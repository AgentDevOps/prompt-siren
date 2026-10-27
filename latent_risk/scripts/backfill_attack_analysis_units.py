#!/usr/bin/env python3
"""Backfill v1 attack-relation analysis units from persisted executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from prompt_siren.attack_relation_classification import build_attack_relation_analysis
from prompt_siren.job.models import (
    TASK_ATTACK_ANALYSIS_FILENAME,
    TASK_EXECUTION_FILENAME,
    TASK_EXECUTION_METADATA_FILENAME,
)

BackfillStatus = Literal["created", "skipped", "would_create", "failed"]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def dump_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _execution_attacks(
    execution: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> Any:
    if metadata is not None and metadata.get("attacks") is not None:
        return metadata["attacks"]
    return execution.get("attacks")


def _validate_metadata_identity(execution: dict[str, Any], metadata: dict[str, Any] | None) -> None:
    if metadata is None:
        return
    for key in ("task_id", "run_id", "execution_id"):
        if metadata.get(key) is not None and metadata[key] != execution.get(key):
            raise ValueError(f"execution_metadata.json {key} does not match execution.json")


def build_backfill_payload(
    execution: dict[str, Any],
    *,
    attacks: Any,
    include_attack_context_in_documents: bool,
) -> dict[str, Any]:
    messages = execution.get("messages")
    if not isinstance(messages, list):
        raise ValueError("execution.json does not contain a messages list")
    for required_key in ("task_id", "run_id", "execution_id", "timestamp"):
        if execution.get(required_key) is None:
            raise ValueError(f"execution.json is missing {required_key}")

    analysis = build_attack_relation_analysis(
        messages,
        attacks=attacks,
        schema_version="v1",
        include_attack_context_in_documents=include_attack_context_in_documents,
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
    trajectory_id = f"{execution['task_id']}/{execution['run_id']}"
    for unit in payload["units"]:
        unit["trajectory_id"] = trajectory_id
    return payload


def backfill_execution(
    execution_path: Path,
    *,
    overwrite: bool = False,
    dry_run: bool = False,
    include_attack_context_in_documents: bool = True,
    update_execution_metadata: bool = False,
) -> tuple[BackfillStatus, str]:
    destination = execution_path.with_name(TASK_ATTACK_ANALYSIS_FILENAME)
    if destination.exists() and not overwrite:
        return "skipped", "analysis file already exists"

    try:
        execution = load_json(execution_path)
        metadata_path = execution_path.with_name(TASK_EXECUTION_METADATA_FILENAME)
        metadata = load_json(metadata_path) if metadata_path.exists() else None
        _validate_metadata_identity(execution, metadata)
        payload = build_backfill_payload(
            execution,
            attacks=_execution_attacks(execution, metadata),
            include_attack_context_in_documents=include_attack_context_in_documents,
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
        return "failed", str(exc)

    if dry_run:
        return (
            "would_create",
            f"{len(payload['units'])} units, attack_context={payload['attack_context_status']}",
        )

    dump_json_atomic(destination, payload)
    if update_execution_metadata:
        if metadata is None:
            metadata = {
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
        metadata_analysis = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "task_id",
                "run_id",
                "execution_id",
                "timestamp",
                "trace_id",
                "span_id",
            }
        }
        metadata_analysis["units"] = [
            {key: value for key, value in unit.items() if key != "trajectory_id"}
            for unit in payload["units"]
        ]
        metadata["attack_analysis"] = metadata_analysis
        dump_json_atomic(metadata_path, metadata)
    return (
        "created",
        f"{len(payload['units'])} units, attack_context={payload['attack_context_status']}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more execution.json files or job directories.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--exclude-attack-context-in-documents",
        action="store_true",
        help="Keep attack metadata but omit its copied text from unit documents.",
    )
    parser.add_argument(
        "--update-execution-metadata",
        action="store_true",
        help="Also add attack_analysis to execution_metadata.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    execution_paths = find_execution_paths(args.inputs)
    if not execution_paths:
        raise SystemExit("No execution.json files found")
    counts: dict[BackfillStatus, int] = {
        "created": 0,
        "skipped": 0,
        "would_create": 0,
        "failed": 0,
    }
    for execution_path in execution_paths:
        status, detail = backfill_execution(
            execution_path,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            include_attack_context_in_documents=(not args.exclude_attack_context_in_documents),
            update_execution_metadata=args.update_execution_metadata,
        )
        counts[status] += 1
        print(f"{status}: {execution_path} ({detail})")
    print("summary:", json.dumps(counts, sort_keys=True))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
