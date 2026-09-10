#!/usr/bin/env python3
"""Extract LLM-judged attack chains from completed execution trajectories."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from prompt_siren.attack_chain_codebook_labeling import label_attack_chain_safely
from prompt_siren.attack_chain_judge import judge_attack_chain, render_attack_chain_markdown
from prompt_siren.attack_chain_topic_retrieval import retrieve_attack_chain_candidates
from prompt_siren.job.models import (
    TASK_ATTACK_CHAIN_JUDGE_FILENAME,
    TASK_ATTACK_CHAIN_JUDGE_MARKDOWN_FILENAME,
    TASK_EXECUTION_FILENAME,
    TASK_EXECUTION_METADATA_FILENAME,
    TASK_RESULT_FILENAME,
)
from prompt_siren.providers import infer_model

ExtractionStatus = Literal["created", "skipped", "would_create", "failed"]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def find_execution_paths(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for input_path in inputs:
        if input_path.is_file() and input_path.name == TASK_EXECUTION_FILENAME:
            paths.add(input_path.resolve())
        elif input_path.is_dir():
            paths.update(path.resolve() for path in input_path.rglob(TASK_EXECUTION_FILENAME))
        else:
            raise FileNotFoundError(f"No execution file or directory found at {input_path}")
    return sorted(paths)


def execution_attacks(execution_path: Path, execution: dict[str, Any]) -> Any:
    metadata_path = execution_path.with_name(TASK_EXECUTION_METADATA_FILENAME)
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        if metadata.get("attacks") is not None:
            return metadata["attacks"]
    attacks = execution.get("attacks")
    if attacks is not None:
        return attacks

    result_path = execution_path.with_name(TASK_RESULT_FILENAME)
    if result_path.exists():
        result = load_json(result_path)
        if result.get("attacks") is not None:
            return result["attacks"]
    return None


def dump_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


async def extract_execution(
    execution_path: Path,
    *,
    model_name: str,
    schema_version: str,
    max_attempts: int,
    overwrite: bool,
    dry_run: bool,
    top_topics_per_group: int = 3,
    top_units_per_group: int = 3,
    min_topic_size: int = 3,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    max_output_tokens: int = 4096,
    recall_priority: bool = False,
    semantic_precision: bool = False,
    codebook_path: Path | None = None,
    codebook_batch_size: int = 8,
) -> tuple[ExtractionStatus, str]:
    json_path = execution_path.with_name(TASK_ATTACK_CHAIN_JUDGE_FILENAME)
    markdown_path = execution_path.with_name(TASK_ATTACK_CHAIN_JUDGE_MARKDOWN_FILENAME)
    if (json_path.exists() or markdown_path.exists()) and not overwrite:
        return "skipped", "attack-chain judge output already exists"
    if dry_run:
        return "would_create", f"{json_path.name} and {markdown_path.name}"

    try:
        execution = load_json(execution_path)
        messages = execution.get("messages")
        if not isinstance(messages, list):
            raise ValueError("execution.json does not contain a messages list")
        attacks = execution_attacks(execution_path, execution)
        try:
            topic_retrieval = await asyncio.to_thread(
                retrieve_attack_chain_candidates,
                messages,
                attacks=attacks,
                top_topics_per_group=top_topics_per_group,
                top_units_per_group=top_units_per_group,
                min_topic_size=min_topic_size,
                embedding_model_name=embedding_model_name,
            )
            candidate_message_indices = topic_retrieval.get("candidate_message_indices") or None
            if candidate_message_indices is None:
                topic_retrieval["fallback"] = "full_trajectory_no_candidates"
        except Exception as retrieval_exc:
            candidate_message_indices = None
            topic_retrieval = {
                "status": "failed",
                "fallback": "full_trajectory",
                "error": f"{type(retrieval_exc).__name__}: {retrieval_exc}",
            }
        model_settings: dict[str, Any] = {"max_tokens": max_output_tokens}
        if "qwen" in model_name.casefold():
            model_settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        analysis = await judge_attack_chain(
            messages,
            attacks=attacks,
            model=infer_model(model_name),
            model_settings=model_settings,
            schema_version=schema_version,
            max_attempts=max_attempts,
            candidate_message_indices=candidate_message_indices,
            topic_retrieval=topic_retrieval,
            recall_priority=recall_priority,
            semantic_precision=semantic_precision,
        )
        payload = {
            key: execution.get(key)
            for key in (
                "task_id",
                "run_id",
                "execution_id",
                "timestamp",
                "trace_id",
                "span_id",
            )
        }
        payload.update(analysis.to_json())
        if codebook_path is not None:
            payload["codebook_labeling"] = await label_attack_chain_safely(
                payload,
                messages,
                codebook_path=codebook_path,
                model=infer_model(model_name),
                model_settings=model_settings,
                batch_size=codebook_batch_size,
                max_attempts=max_attempts,
            )
        dump_text_atomic(
            json_path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )
        dump_text_atomic(markdown_path, render_attack_chain_markdown(payload, messages))
        if codebook_path is not None and payload["codebook_labeling"]["status"] == "failed":
            return "failed", (
                "attack chain saved, but codebook labeling failed: "
                + payload["codebook_labeling"]["error"]
            )
    except Exception as exc:
        return "failed", str(exc)
    return (
        "created",
        f"chain_observed={analysis.chain_observed}, "
        f"messages={len(analysis.attack_message_indices)}, nodes={len(analysis.nodes)}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more execution.json files or completed job directories.",
    )
    parser.add_argument("--model", required=True, help="PydanticAI-compatible judge model name.")
    parser.add_argument("--schema-version", default="v2")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--top-topics-per-group", type=int, default=3)
    parser.add_argument("--top-units-per-group", type=int, default=3)
    parser.add_argument("--min-topic-size", type=int, default=3)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--codebook", type=Path, help="Label the extracted chain using this Markdown codebook."
    )
    parser.add_argument("--codebook-batch-size", type=int, default=8)
    judge_mode = parser.add_mutually_exclusive_group()
    judge_mode.add_argument(
        "--recall-priority",
        action="store_true",
        help=(
            "Use the over-inclusive open-coding prompt with per-evidence role allowlists; "
            "the current compact mode remains the default."
        ),
    )
    judge_mode.add_argument(
        "--semantic-precision",
        action="store_true",
        help="Use a slightly tighter causal prompt and action-only tool expansion.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    execution_paths = find_execution_paths(args.inputs)
    if not execution_paths:
        raise SystemExit("No execution.json files found")
    counts: dict[ExtractionStatus, int] = {
        "created": 0,
        "skipped": 0,
        "would_create": 0,
        "failed": 0,
    }
    for execution_path in execution_paths:
        status, detail = await extract_execution(
            execution_path,
            model_name=args.model,
            schema_version=args.schema_version,
            max_attempts=max(1, args.max_attempts),
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            top_topics_per_group=max(1, args.top_topics_per_group),
            top_units_per_group=max(0, args.top_units_per_group),
            min_topic_size=max(2, args.min_topic_size),
            embedding_model_name=args.embedding_model,
            max_output_tokens=max(1, args.max_output_tokens),
            recall_priority=args.recall_priority,
            semantic_precision=args.semantic_precision,
            codebook_path=args.codebook,
            codebook_batch_size=max(1, args.codebook_batch_size),
        )
        counts[status] += 1
        print(f"{status}: {execution_path} ({detail})")
    print("summary:", json.dumps(counts, sort_keys=True))
    if counts["failed"]:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
