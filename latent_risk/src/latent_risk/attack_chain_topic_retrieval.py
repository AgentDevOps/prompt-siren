# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Retrieve payload-near messages from one trajectory with BERTopic."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pydantic_ai.messages import ModelMessage

from .attack_relation_classification import (
    build_attack_relation_analysis,
    normalize_document,
)
from .trajectory_labeling import messages_to_dicts, serializable_attacks
from .types import InjectionAttack

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VERBATIM_PAYLOAD_MIN_CHARS = 32
STRUCTURAL_STOP_WORDS = {
    "action",
    "agent_message",
    "assistant",
    "next",
    "observation",
    "observation_to_agent_transition",
    "result_observed",
    "thought",
    "tool",
    "tool_call_id",
    "tool_outcome",
    "unit",
    "unknown",
}


@dataclass(frozen=True)
class _RetrievalUnit:
    """Minimal source-separated view used only for attack-chain retrieval."""

    unit_type: str
    raw_document: str
    normalized_document: str
    message_indices: list[int]


def _canonical_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _payload_texts(
    attacks: Mapping[str, InjectionAttack] | dict[str, Any] | None,
) -> list[str]:
    serialized = serializable_attacks(attacks) or {}
    return [
        attack["content"].strip()
        for attack in serialized.values()
        if isinstance(attack, dict)
        and isinstance(attack.get("content"), str)
        and attack["content"].strip()
    ]


def _payload_exposure_indices(
    messages: Sequence[ModelMessage | dict[str, Any]], payloads: Sequence[str]
) -> list[int]:
    serialized_messages = messages_to_dicts(messages)
    indices: set[int] = set()
    for payload in (_canonical_text(payload) for payload in payloads if payload.strip()):
        for message_index, message in enumerate(serialized_messages):
            contents = [
                part.get("content")
                for part in message.get("parts", [])
                if isinstance(part, dict) and isinstance(part.get("content"), str)
            ]
            if any(payload in _canonical_text(content) for content in contents):
                indices.add(message_index)
                break
    return sorted(indices)


def _source_separated_retrieval_units(
    analysis: Any,
) -> list[Any]:
    """Embed source-separated messages, but never semantic transition units."""
    source_events = getattr(analysis, "source_events", None)
    non_transition_units = [
        unit
        for unit in analysis.units
        if unit.unit_type != "observation_to_agent_transition"
    ]
    if not isinstance(source_events, list):
        return non_transition_units

    units = [unit for unit in non_transition_units if unit.unit_type != "agent_message"]
    for event in source_events:
        if event.source_type == "assistant-intent":
            source = "assistant-text"
        elif event.source_type == "assistant-action":
            source = "tool-call:" + (event.tool_name or "unknown")
        else:
            continue
        raw_document = (
            "[UNIT=AGENT_MESSAGE]\n"
            f"[SOURCE={source} part_index={event.part_index}]\n"
            f"{event.content}"
        )
        units.append(
            _RetrievalUnit(
                unit_type="agent_message",
                raw_document=raw_document,
                normalized_document=normalize_document(raw_document),
                message_indices=[event.message_index],
            )
        )
    return units


def _deterministic_next_assistant_indices(
    messages: Sequence[ModelMessage | dict[str, Any]],
    source_message_indices: set[int],
) -> list[int]:
    """Return the next assistant message after each selected observation/exposure."""
    serialized_messages = messages_to_dicts(messages)
    selected: set[int] = set()
    for source_index in source_message_indices:
        for message_index in range(source_index + 1, len(serialized_messages)):
            message = serialized_messages[message_index]
            if message.get("kind") != "response":
                continue
            if any(
                isinstance(part, dict)
                and part.get("part_kind") in {"text", "tool-call"}
                for part in message.get("parts", [])
            ):
                selected.add(message_index)
                break
    return sorted(selected)


def _tool_return_message_indices(
    messages: Sequence[ModelMessage | dict[str, Any]],
) -> set[int]:
    return {
        message_index
        for message_index, message in enumerate(messages_to_dicts(messages))
        if any(
            isinstance(part, dict)
            and part.get("part_kind") in {"tool-return", "injectable-tool-return"}
            for part in message.get("parts", [])
        )
    }


def _nested_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for nested in value.values() for text in _nested_strings(nested)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [text for nested in value for text in _nested_strings(nested)]
    return []


def _message_payload_flags(
    messages: Sequence[ModelMessage | dict[str, Any]],
    canonical_payloads: Sequence[str],
) -> dict[int, bool]:
    """Report which complete messages contain a full verbatim payload."""
    flags: dict[int, bool] = {}
    for message_index, message in enumerate(messages_to_dicts(messages)):
        values = [
            text
            for part in message.get("parts", [])
            if isinstance(part, dict)
            for value in (part.get("content"), part.get("args"))
            for text in _nested_strings(value)
        ]
        canonical_values = [_canonical_text(value) for value in values]
        flags[message_index] = any(
            payload in value for payload in canonical_payloads for value in canonical_values
        )
    return flags


def _paired_tool_return_indices(
    messages: Sequence[ModelMessage | dict[str, Any]],
    call_message_indices: set[int],
) -> list[int]:
    """Return tool results linked to any candidate assistant tool call."""
    serialized_messages = messages_to_dicts(messages)
    call_ids = {
        str(part["tool_call_id"])
        for message_index, message in enumerate(serialized_messages)
        if message_index in call_message_indices
        for part in message.get("parts", [])
        if isinstance(part, dict)
        and part.get("part_kind") == "tool-call"
        and part.get("tool_call_id")
    }
    return sorted(
        {
            message_index
            for message_index, message in enumerate(serialized_messages)
            for part in message.get("parts", [])
            if isinstance(part, dict)
            and part.get("part_kind") in {"tool-return", "injectable-tool-return"}
            and str(part.get("tool_call_id") or "") in call_ids
        }
    )


def _payload_command_fragments(payloads: Sequence[str]) -> set[str]:
    """Return substantial payload lines that can be matched in tool arguments."""
    fragments: set[str] = set()
    for payload in payloads:
        code_fragments = re.findall(r"(?<!`)`([^`\n]+)`(?!`)", payload)
        fenced_blocks = re.findall(
            r"```[^\n]*\n(.*?)```",
            payload,
            flags=re.DOTALL,
        )
        code_fragments.extend(
            line for block in fenced_blocks for line in block.splitlines() if line.strip()
        )
        for fragment in code_fragments:
            canonical = _canonical_text(fragment)
            if len(canonical) >= 8:
                fragments.add(canonical)
        for line in payload.splitlines():
            canonical = _canonical_text(line.strip().strip("`"))
            if len(canonical) >= 16:
                fragments.add(canonical)
    return fragments


def _payload_matching_tool_messages(
    messages: Sequence[ModelMessage | dict[str, Any]], payloads: Sequence[str]
) -> tuple[list[int], list[int]]:
    """Find tool calls reproducing payload text and their call-id-linked returns."""
    serialized_messages = messages_to_dicts(messages)
    fragments = _payload_command_fragments(payloads)
    matching_call_indices: set[int] = set()
    matching_call_ids: set[str] = set()
    for message_index, message in enumerate(serialized_messages):
        for part in message.get("parts", []):
            if not isinstance(part, dict) or part.get("part_kind") != "tool-call":
                continue
            argument_strings = [_canonical_text(text) for text in _nested_strings(part.get("args"))]
            if not any(
                fragment in argument for argument in argument_strings for fragment in fragments
            ):
                continue
            matching_call_indices.add(message_index)
            call_id = part.get("tool_call_id")
            if isinstance(call_id, str) and call_id:
                matching_call_ids.add(call_id)

    paired_return_indices: set[int] = set()
    for message_index, message in enumerate(serialized_messages):
        for part in message.get("parts", []):
            if (
                isinstance(part, dict)
                and part.get("part_kind") in {"tool-return", "injectable-tool-return"}
                and part.get("tool_call_id") in matching_call_ids
            ):
                paired_return_indices.add(message_index)
    return sorted(matching_call_indices), sorted(paired_return_indices)


@lru_cache(maxsize=4)
def _embedding_model(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _cosine(left: Any, right: Any) -> float:
    import numpy as np

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left_array) * np.linalg.norm(right_array)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def _behavior_document(document: str) -> str:
    marker_index = document.find("[UNIT=")
    return document[marker_index:].strip() if marker_index >= 0 else document.strip()


def retrieve_attack_chain_candidates(
    messages: Sequence[ModelMessage | dict[str, Any]],
    *,
    attacks: Mapping[str, InjectionAttack] | dict[str, Any] | None,
    top_topics_per_group: int = 3,
    top_units_per_group: int = 3,
    min_topic_size: int = 3,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Cluster one trajectory and return message indices from payload-near topics."""
    try:
        import numpy as np
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
        from umap import UMAP
    except ImportError as exc:
        raise RuntimeError(
            "BERTopic retrieval requires the 'topics' extra: uv sync --extra topics"
        ) from exc

    payloads = _payload_texts(attacks)
    if not payloads:
        return {
            "status": "skipped",
            "reason": "attack payload unavailable",
            "groups": {},
            "candidate_message_indices": [],
            "payload_exposure_message_indices": [],
            "payload_matching_tool_call_message_indices": [],
            "paired_tool_return_message_indices": [],
        }

    analysis = build_attack_relation_analysis(
        messages,
        attacks=attacks,
        include_attack_context_in_documents=False,
    )
    retrieval_units = _source_separated_retrieval_units(analysis)
    grouped_units: dict[str, list[Any]] = defaultdict(list)
    canonical_payloads = [
        canonical
        for payload in payloads
        if len(canonical := _canonical_text(payload)) >= VERBATIM_PAYLOAD_MIN_CHARS
    ]
    excluded_verbatim_units = 0
    payload_message_flags = _message_payload_flags(messages, canonical_payloads)
    verbatim_neighbor_indices: set[int] = set()
    for unit in retrieval_units:
        canonical_document = _canonical_text(_behavior_document(unit.raw_document))
        if any(payload in canonical_document for payload in canonical_payloads):
            excluded_verbatim_units += 1
            verbatim_neighbor_indices.update(
                message_index
                for message_index in unit.message_indices
                if not payload_message_flags.get(message_index, False)
            )
            continue
        grouped_units[unit.unit_type].append(unit)

    embedding_model = _embedding_model(embedding_model_name)
    payload_document = "\n\n---\n\n".join(payloads)
    stop_words = sorted(set(ENGLISH_STOP_WORDS) | STRUCTURAL_STOP_WORDS)
    selected_message_indices: set[int] = set()
    groups: dict[str, Any] = {}

    for group_name, units in sorted(grouped_units.items()):
        documents = [_behavior_document(unit.normalized_document) for unit in units]
        required_documents = max(3, min_topic_size)
        if len(documents) < required_documents:
            groups[group_name] = {
                "status": "skipped",
                "reason": f"requires at least {required_documents} documents",
                "document_count": len(documents),
            }
            continue

        embeddings = embedding_model.encode([*documents, payload_document], show_progress_bar=False)
        behavior_embeddings = embeddings[:-1]
        payload_embedding = embeddings[-1]
        model = BERTopic(
            embedding_model=embedding_model,
            vectorizer_model=CountVectorizer(stop_words=stop_words),
            umap_model=UMAP(
                n_neighbors=min(10, len(documents) - 1),
                n_components=min(5, len(documents) - 2),
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            ),
            hdbscan_model=HDBSCAN(
                min_cluster_size=max(2, min(min_topic_size, len(documents))),
                min_samples=1,
                metric="euclidean",
                cluster_selection_method="leaf",
                prediction_data=True,
            ),
            min_topic_size=max(2, min(min_topic_size, len(documents))),
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = model.fit_transform(documents, embeddings=behavior_embeddings)
        similarities = [_cosine(embedding, payload_embedding) for embedding in behavior_embeddings]
        topic_rows: dict[int, list[int]] = defaultdict(list)
        for unit_index, topic_id in enumerate(topics):
            topic_rows[int(topic_id)].append(unit_index)

        ranked_topics: list[dict[str, Any]] = []
        for topic_id, unit_indices in topic_rows.items():
            values = np.asarray([similarities[index] for index in unit_indices], dtype=float)
            ranked_topics.append(
                {
                    "topic_id": topic_id,
                    "unit_count": len(unit_indices),
                    "payload_similarity_p90": float(np.percentile(values, 90)),
                    "payload_similarity_max": float(np.max(values)),
                    "unit_indices": unit_indices,
                }
            )
        non_outliers = [topic for topic in ranked_topics if topic["topic_id"] != -1]
        selected_topics = sorted(
            non_outliers,
            key=lambda topic: (
                topic["payload_similarity_p90"],
                topic["payload_similarity_max"],
            ),
            reverse=True,
        )[: max(1, top_topics_per_group)]

        compact_topics: list[dict[str, Any]] = []
        for rank, topic in enumerate(selected_topics, start=1):
            message_indices = sorted(
                {
                    message_index
                    for unit_index in topic.pop("unit_indices")
                    for message_index in units[unit_index].message_indices
                }
            )
            selected_message_indices.update(message_indices)
            compact_topics.append({"rank": rank, **topic, "message_indices": message_indices})
        groups[group_name] = {
            "status": "fitted",
            "document_count": len(documents),
            "selected_topics": compact_topics,
        }

        unit_rows = [
            {
                "unit_index": unit_index,
                "topic_id": int(topics[unit_index]),
                "payload_similarity": similarities[unit_index],
                "message_indices": sorted(set(unit.message_indices)),
            }
            for unit_index, unit in enumerate(units)
        ]
        individual_limit = max(0, top_units_per_group)
        selected_individuals = sorted(
            unit_rows,
            key=lambda row: row["payload_similarity"],
            reverse=True,
        )[:individual_limit]
        selected_outliers = sorted(
            (row for row in unit_rows if row["topic_id"] == -1),
            key=lambda row: row["payload_similarity"],
            reverse=True,
        )[:individual_limit]
        safety_net_rows: list[dict[str, Any]] = []
        seen_unit_indices: set[int] = set()
        for selection_source, rows in (
            ("top_individual", selected_individuals),
            ("top_outlier", selected_outliers),
        ):
            for row in rows:
                unit_index = row["unit_index"]
                if unit_index in seen_unit_indices:
                    continue
                seen_unit_indices.add(unit_index)
                selected_message_indices.update(row["message_indices"])
                safety_net_rows.append({"selection_source": selection_source, **row})
        groups[group_name]["individual_safety_net"] = safety_net_rows

    exposure_indices = _payload_exposure_indices(messages, payloads)
    matching_tool_call_indices, paired_tool_return_indices = _payload_matching_tool_messages(
        messages, payloads
    )
    selected_message_indices.update(exposure_indices)
    selected_message_indices.update(matching_tool_call_indices)
    selected_message_indices.update(paired_tool_return_indices)
    selected_message_indices.update(verbatim_neighbor_indices)
    selected_observation_indices = (
        selected_message_indices & _tool_return_message_indices(messages)
    )
    deterministic_transition_neighbor_indices = _deterministic_next_assistant_indices(
        messages,
        set(exposure_indices)
        | selected_observation_indices
        | set(paired_tool_return_indices),
    )
    selected_message_indices.update(deterministic_transition_neighbor_indices)
    candidate_paired_return_indices = _paired_tool_return_indices(
        messages,
        selected_message_indices,
    )
    selected_message_indices.update(candidate_paired_return_indices)
    return {
        "status": "fitted",
        "method": "single_trajectory_bertopic_payload_retrieval",
        "embedding_input": "behavior_only",
        "attack_similarity_metric": "cosine(payload_only, behavior_only)",
        "topic_ranking_metric": "p90",
        "embedding_model": embedding_model_name,
        "top_topics_per_group": max(1, top_topics_per_group),
        "top_units_per_group": max(0, top_units_per_group),
        "min_topic_size": max(2, min_topic_size),
        "verbatim_payload_units_excluded": excluded_verbatim_units,
        "agent_embedding_views": "assistant_text_and_tool_call_separate",
        "transition_semantic_retrieval": False,
        "groups": groups,
        "payload_exposure_message_indices": exposure_indices,
        "verbatim_payload_neighbor_message_indices": sorted(verbatim_neighbor_indices),
        "deterministic_transition_neighbor_message_indices": (
            deterministic_transition_neighbor_indices
        ),
        "payload_matching_tool_call_message_indices": matching_tool_call_indices,
        "paired_tool_return_message_indices": paired_tool_return_indices,
        "candidate_paired_tool_return_message_indices": candidate_paired_return_indices,
        "candidate_message_indices": sorted(selected_message_indices),
    }
