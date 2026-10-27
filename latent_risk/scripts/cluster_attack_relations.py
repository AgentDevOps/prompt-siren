#!/usr/bin/env python3
"""Discover behavior topics and rank them by payload proximity."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from prompt_siren.attack_relation_classification import normalize_document

ANALYSIS_FILENAME = "attack_analysis_units.json"
DocumentView = Literal["raw", "normalized"]
EmbeddingConditioning = Literal["behavior_only", "attack_context_plus_behavior"]
DEFAULT_VERBATIM_PAYLOAD_MIN_CHARS = 32
STRUCTURAL_STOP_WORDS = {
    "abs_path",
    "abspath",
    "action",
    "agent_message",
    "assistant",
    "attack",
    "context",
    "hex_id",
    "next",
    "observation",
    "observation_to_agent_transition",
    "result_observed",
    "thought",
    "timestamp",
    "tool",
    "tool_call_id",
    "tool_outcome",
    "tool_succeeded",
    "unit",
    "unknown",
    "url",
    "uuid",
}


def _group_name(unit: dict[str, Any], *, group_by_tool: bool) -> str:
    unit_type = str(unit.get("unit_type", "unknown"))
    if not group_by_tool:
        return unit_type
    tool_name = unit.get("tool_name")
    return f"{unit_type}:{tool_name or 'no-tool'}"


def _behavior_document(conditioned_document: str) -> str:
    """Remove the repeated attack prefix for c-TF-IDF topic representation."""
    marker = "[UNIT="
    marker_index = conditioned_document.find(marker)
    if marker_index < 0:
        return conditioned_document
    return conditioned_document[marker_index:].strip()


def _attack_documents(payload: dict[str, Any]) -> list[str]:
    """Extract each textual payload without copying it into persisted assignments."""
    sections: list[str] = []
    for item in payload.get("attack_context", []):
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and content.strip():
            sections.append(content.strip())
    return sections


def _attack_document(payload: dict[str, Any]) -> str | None:
    sections = _attack_documents(payload)
    return "\n\n---\n\n".join(sections) or None


def collect_documents(
    job_dirs: Path | Sequence[Path],
    *,
    document_view: DocumentView,
    group_by_tool: bool = False,
    reference_root: Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Collect every unlabeled unit; no relation-based filtering is applied."""
    input_dirs = [job_dirs] if isinstance(job_dirs, Path) else list(job_dirs)
    resolved_dirs = [path.resolve() for path in input_dirs]
    if not resolved_dirs:
        raise ValueError("At least one job directory is required")
    for path in resolved_dirs:
        if not path.is_dir():
            raise FileNotFoundError(f"Job directory not found: {path}")
    resolved_root = (reference_root or common_reference_root(resolved_dirs)).resolve()
    document_key = f"{document_view}_document"
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    analysis_paths = sorted(
        {path.resolve() for job_dir in resolved_dirs for path in job_dir.rglob(ANALYSIS_FILENAME)}
    )
    for path in analysis_paths:
        payload = json.loads(path.read_text())
        attack_documents = _attack_documents(payload)
        attack_document = _attack_document(payload)
        for unit in payload.get("units", []):
            if not isinstance(unit, dict):
                continue
            document = unit.get(document_key)
            if not isinstance(document, str) or not document.strip():
                continue
            message_indices = unit.get("message_indices", [])
            from_message_id = unit.get("from_message_id")
            to_message_id = unit.get("to_message_id")
            if unit.get("unit_type") == "observation_to_agent_transition":
                if from_message_id is None and message_indices:
                    from_message_id = f"m{message_indices[0]}"
                if to_message_id is None and len(message_indices) > 1:
                    to_message_id = f"m{message_indices[-1]}"
            groups[_group_name(unit, group_by_tool=group_by_tool)].append(
                {
                    "embedding_document": document,
                    "topic_document": _behavior_document(document),
                    "attack_document": attack_document,
                    "attack_documents": attack_documents,
                    "unit_id": unit.get("unit_id"),
                    "unit_type": unit.get("unit_type"),
                    "trajectory_id": unit.get("trajectory_id"),
                    "task_id": payload.get("task_id"),
                    "run_id": payload.get("run_id"),
                    "group_id": unit.get("group_id"),
                    "source_event_ids": unit.get("source_event_ids", []),
                    "message_indices": message_indices,
                    "from_message_id": from_message_id,
                    "to_message_id": to_message_id,
                    "tool_name": unit.get("tool_name"),
                    "tool_call_id": unit.get("tool_call_id"),
                    "distance": unit.get("distance"),
                    "observation_event_id": unit.get("observation_event_id"),
                    "next_assistant_event_id": unit.get("next_assistant_event_id"),
                    "attack_context_status": unit.get("attack_context_status"),
                    "source_file": str(path.relative_to(resolved_root)),
                    "source_job_dir": str(
                        next(
                            job_dir.relative_to(resolved_root)
                            for job_dir in resolved_dirs
                            if path.is_relative_to(job_dir)
                        )
                    ),
                }
            )
    return dict(groups)


def common_reference_root(job_dirs: Sequence[Path]) -> Path:
    """Return a stable root from which all source_file references are relative."""
    resolved_dirs = [path.resolve() for path in job_dirs]
    if not resolved_dirs:
        raise ValueError("At least one job directory is required")
    if len(resolved_dirs) == 1:
        return resolved_dirs[0]
    return Path(os.path.commonpath([str(path) for path in resolved_dirs]))


def _canonical_match_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _verbatim_payload_match_length(
    record: dict[str, Any],
    *,
    document_view: DocumentView,
    min_chars: int,
) -> int | None:
    behavior = record.get("topic_document")
    if not isinstance(behavior, str):
        return None
    canonical_behavior = _canonical_match_text(behavior)
    for payload in record.get("attack_documents", []):
        if not isinstance(payload, str):
            continue
        comparable_payload = (
            normalize_document(payload) if document_view == "normalized" else payload
        )
        canonical_payload = _canonical_match_text(comparable_payload)
        if len(canonical_payload) >= min_chars and canonical_payload in canonical_behavior:
            return len(canonical_payload)
    return None


def filter_verbatim_payload_units(
    groups: dict[str, list[dict[str, Any]]],
    *,
    document_view: DocumentView,
    enabled: bool = True,
    min_chars: int = DEFAULT_VERBATIM_PAYLOAD_MIN_CHARS,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Exclude units whose behavior contains a complete textual attack payload."""
    kept: dict[str, list[dict[str, Any]]] = {}
    excluded: list[dict[str, Any]] = []
    excluded_by_group: dict[str, int] = {}
    for group_name, records in groups.items():
        kept_records: list[dict[str, Any]] = []
        for record in records:
            match_length = (
                _verbatim_payload_match_length(
                    record,
                    document_view=document_view,
                    min_chars=min_chars,
                )
                if enabled
                else None
            )
            if match_length is None:
                kept_records.append(record)
                continue
            excluded_by_group[group_name] = excluded_by_group.get(group_name, 0) + 1
            excluded.append(
                {
                    **_sample_record(record),
                    "group": group_name,
                    "exclusion_reason": "behavior_contains_complete_payload",
                    "matched_payload_chars": match_length,
                }
            )
        if kept_records:
            kept[group_name] = kept_records
    return kept, {
        "enabled": enabled,
        "policy": "exclude_behavior_containing_complete_payload",
        "comparison": f"{document_view}_payload_substring_in_behavior_only",
        "min_payload_chars": min_chars,
        "excluded_unit_count": len(excluded),
        "excluded_by_group": excluded_by_group,
        "excluded_units": excluded,
    }


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _membership_probability(probabilities: Any, index: int) -> float | None:
    if probabilities is None:
        return None
    row = probabilities[index]
    if hasattr(row, "max"):
        return float(row.max())
    if isinstance(row, int | float):
        return float(row)
    return None


def _sample_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        in {
            "unit_id",
            "unit_type",
            "trajectory_id",
            "task_id",
            "run_id",
            "group_id",
            "source_event_ids",
            "message_indices",
            "from_message_id",
            "to_message_id",
            "tool_name",
            "tool_call_id",
            "observation_event_id",
            "next_assistant_event_id",
            "source_file",
            "source_job_dir",
            "topic_id",
            "topic_namespace",
            "topic_membership_probability",
            "attack_similarity",
        }
    }


def _compact_topic_info(topic_info: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop BERTopic's copied representative documents from persisted output."""
    return [
        {key: value for key, value in topic.items() if key != "Representative_Docs"}
        for topic in topic_info
    ]


def _cosine_similarity(left: Any, right: Any) -> float:
    import numpy as np

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def _attack_similarity_summary(values: list[float]) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    top_values = np.sort(array)[-min(5, len(array)) :]
    return {
        "unit_count": len(values),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "top5_mean": float(np.mean(top_values)),
    }


def _rank_topics_by_attack_similarity(
    topic_info: list[dict[str, Any]],
    assignments: list[dict[str, Any]],
    *,
    group_name: str,
) -> list[dict[str, Any]]:
    values_by_topic: dict[int, list[float]] = defaultdict(list)
    for assignment in assignments:
        similarity = assignment.get("attack_similarity")
        if isinstance(similarity, int | float):
            values_by_topic[int(assignment["topic_id"])].append(float(similarity))

    info_by_topic = {int(topic["Topic"]): topic for topic in topic_info}
    for topic_id, values in values_by_topic.items():
        if topic_id in info_by_topic:
            info_by_topic[topic_id]["attack_similarity"] = _attack_similarity_summary(values)

    ranked = sorted(
        (
            topic
            for topic in topic_info
            if int(topic.get("Topic", -1)) != -1 and "attack_similarity" in topic
        ),
        key=lambda topic: topic["attack_similarity"]["p90"],
        reverse=True,
    )
    candidates: list[dict[str, Any]] = []
    for rank, topic in enumerate(ranked, start=1):
        topic["attack_proximity_rank"] = rank
        candidates.append(
            {
                "rank": rank,
                "topic_id": int(topic["Topic"]),
                "topic_namespace": f"{group_name}/topic_{int(topic['Topic'])}",
                "name": topic.get("Name"),
                "count": topic.get("Count"),
                "ranking_metric": "p90",
                "ranking_score": topic["attack_similarity"]["p90"],
                "attack_similarity": topic["attack_similarity"],
            }
        )
    return candidates


def build_review_samples(
    assignments: list[dict[str, Any]],
    *,
    samples_per_bucket: int,
    random_seed: int,
) -> dict[str, Any]:
    """Export representative, boundary, random, and outlier units for human review."""
    rng = random.Random(random_seed)
    by_topic: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for assignment in assignments:
        by_topic[int(assignment["topic_id"])].append(assignment)

    topics: dict[str, Any] = {}
    for topic_id, records in sorted(by_topic.items()):
        if topic_id == -1:
            continue
        ranked = sorted(
            records,
            key=lambda record: record.get("topic_membership_probability") or 0.0,
            reverse=True,
        )
        random_records = rng.sample(records, min(samples_per_bucket, len(records)))
        topics[str(topic_id)] = {
            "representative": [_sample_record(record) for record in ranked[:samples_per_bucket]],
            "boundary": [_sample_record(record) for record in ranked[-samples_per_bucket:]],
            "random": [_sample_record(record) for record in random_records],
        }

    outliers = by_topic.get(-1, [])
    return {
        "topics": topics,
        "outliers": [
            _sample_record(record)
            for record in rng.sample(outliers, min(samples_per_bucket, len(outliers)))
        ],
    }


def fit_topics(
    groups: dict[str, list[dict[str, Any]]],
    *,
    document_view: DocumentView,
    min_documents: int,
    min_topic_size: int,
    model_dir: Path | None,
    samples_per_bucket: int,
    random_seed: int,
    embedding_model_name: str,
    embedding_conditioning: EmbeddingConditioning = "behavior_only",
) -> dict[str, Any]:
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    except ImportError as exc:
        raise SystemExit(
            "BERTopic is not installed. Install it with `uv sync --extra topics`."
        ) from exc

    embedding_model = SentenceTransformer(embedding_model_name)
    topic_stop_words = sorted(set(ENGLISH_STOP_WORDS) | STRUCTURAL_STOP_WORDS)

    output: dict[str, Any] = {"groups": {}}
    for group_name, records in sorted(groups.items()):
        if len(records) < min_documents:
            output["groups"][group_name] = {
                "status": "skipped",
                "reason": f"requires at least {min_documents} documents",
                "document_count": len(records),
                "pooled_fallback": "rerun without --group-by-tool",
            }
            continue

        embedding_documents = [record["embedding_document"] for record in records]
        topic_documents = [record["topic_document"] for record in records]
        attack_documents = [record.get("attack_document") for record in records]
        unique_attack_documents = list(
            dict.fromkeys(
                document for document in attack_documents if isinstance(document, str) and document
            )
        )
        if embedding_conditioning == "behavior_only":
            all_documents = [*topic_documents, *unique_attack_documents]
            attack_embedding_offset = len(topic_documents)
        else:
            all_documents = [
                *embedding_documents,
                *topic_documents,
                *unique_attack_documents,
            ]
            attack_embedding_offset = len(embedding_documents) + len(topic_documents)
        all_embeddings = embedding_model.encode(
            all_documents,
            show_progress_bar=True,
        )
        document_count = len(records)
        if embedding_conditioning == "behavior_only":
            behavior_embeddings = all_embeddings[:document_count]
            embeddings = behavior_embeddings
        else:
            embeddings = all_embeddings[:document_count]
            behavior_embeddings = all_embeddings[document_count : 2 * document_count]
        attack_embeddings = {
            document: all_embeddings[attack_embedding_offset + index]
            for index, document in enumerate(unique_attack_documents)
        }
        attack_similarities = [
            (
                _cosine_similarity(behavior_embedding, attack_embeddings[attack_document])
                if isinstance(attack_document, str) and attack_document in attack_embeddings
                else None
            )
            for behavior_embedding, attack_document in zip(
                behavior_embeddings,
                attack_documents,
                strict=True,
            )
        ]
        model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=CountVectorizer(stop_words=topic_stop_words),
            min_topic_size=max(2, min(min_topic_size, len(topic_documents))),
            calculate_probabilities=True,
            verbose=False,
        )
        topics, probabilities = model.fit_transform(topic_documents, embeddings=embeddings)
        topic_info = _compact_topic_info(model.get_topic_info().to_dict(orient="records"))
        model_id = f"{group_name.replace(':', '__')}_{document_view}_{embedding_conditioning}_v3"
        assignments: list[dict[str, Any]] = []
        for index, (record, topic, attack_similarity) in enumerate(
            zip(records, topics, attack_similarities, strict=True)
        ):
            topic_id = int(topic)
            assignments.append(
                {
                    **{
                        key: value
                        for key, value in record.items()
                        if key
                        not in {
                            "embedding_document",
                            "topic_document",
                            "attack_document",
                            "attack_documents",
                        }
                    },
                    "model_id": model_id,
                    "topic_id": topic_id,
                    "topic_namespace": f"{group_name}/topic_{topic_id}",
                    "topic_membership_probability": _membership_probability(probabilities, index),
                    "attack_similarity": attack_similarity,
                    "status": "provisional",
                    "human_review_status": "unreviewed",
                }
            )

        attack_candidate_topics = _rank_topics_by_attack_similarity(
            topic_info,
            assignments,
            group_name=group_name,
        )
        output["groups"][group_name] = {
            "status": "fitted",
            "model_id": model_id,
            "document_view": document_view,
            "embedding_input": embedding_conditioning,
            "topic_representation_input": "behavior_only",
            "embedding_model": embedding_model_name,
            "attack_similarity_metric": "cosine(payload_only, behavior_only)",
            "attack_topic_ranking_metric": "p90",
            "document_count": len(topic_documents),
            "topics": topic_info,
            "attack_candidate_topics": attack_candidate_topics,
            "assignments": assignments,
            "review_samples": build_review_samples(
                assignments,
                samples_per_bucket=samples_per_bucket,
                random_seed=random_seed,
            ),
        }
        if model_dir is not None:
            destination = model_dir / model_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            model.save(destination, serialization="safetensors", save_ctfidf=True)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "job_dirs",
        nargs="+",
        type=Path,
        help="One or more job directories to cluster together.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--document-view", choices=("raw", "normalized"), default="normalized")
    parser.add_argument(
        "--group-by-tool",
        action="store_true",
        help="Split unit types by tool; pooled unit-type models are the default.",
    )
    parser.add_argument("--min-documents", type=int, default=10)
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--samples-per-bucket", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument(
        "--embedding-conditioning",
        choices=("behavior-only", "attack-context-plus-behavior"),
        default="behavior-only",
        help=(
            "Text used for UMAP/HDBSCAN embeddings. Behavior-only is the default; "
            "the attack-context option is retained for ablation."
        ),
    )
    parser.add_argument(
        "--include-verbatim-payload-units",
        action="store_true",
        help=(
            "Disable the default filter that removes units whose behavior contains "
            "a complete payload."
        ),
    )
    parser.add_argument(
        "--verbatim-payload-min-chars",
        type=int,
        default=DEFAULT_VERBATIM_PAYLOAD_MIN_CHARS,
        help="Minimum canonical payload length eligible for complete-payload filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.job_dirs) > 1 and args.output is None:
        raise SystemExit("--output is required when clustering multiple job directories")
    reference_root = common_reference_root(args.job_dirs)
    collected_groups = collect_documents(
        args.job_dirs,
        document_view=args.document_view,
        group_by_tool=args.group_by_tool,
        reference_root=reference_root,
    )
    groups, verbatim_payload_filter = filter_verbatim_payload_units(
        collected_groups,
        document_view=args.document_view,
        enabled=not args.include_verbatim_payload_units,
        min_chars=max(1, args.verbatim_payload_min_chars),
    )
    result = {
        "job_dir": str(reference_root),
        "input_job_dirs": [str(path.resolve()) for path in args.job_dirs],
        "output_schema_version": "v3",
        "document_storage": "referenced",
        "embedding_conditioning": args.embedding_conditioning.replace("-", "_"),
        "topic_representation": "behavior_only",
        "attack_similarity_metric": "cosine(payload_only, behavior_only)",
        "discovery_stage": "provisional_topics",
        "document_view": args.document_view,
        "group_by_tool": args.group_by_tool,
        "collected_document_count": sum(len(records) for records in collected_groups.values()),
        "input_document_count": sum(len(records) for records in groups.values()),
        "verbatim_payload_filter": verbatim_payload_filter,
        **fit_topics(
            groups,
            document_view=args.document_view,
            min_documents=args.min_documents,
            min_topic_size=args.min_topic_size,
            model_dir=args.model_dir,
            samples_per_bucket=args.samples_per_bucket,
            random_seed=args.random_seed,
            embedding_model_name=args.embedding_model,
            embedding_conditioning=args.embedding_conditioning.replace("-", "_"),
        ),
    }
    destination = args.output or args.job_dirs[0] / "attack_relation_topics.json"
    destination.write_text(json.dumps(result, indent=2, default=_json_default) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
