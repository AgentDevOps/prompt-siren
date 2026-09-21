#!/usr/bin/env python3
"""Attribute an already extracted attack chain to a primary replay candidate.

Every run (including a single trajectory) also writes a short Markdown summary
listing which trajectories have ranked replay candidates and which don't.

Example:
    uv run python scripts/attribute_attack_chains.py jobs/my_job \\
        --model MODEL_NAME --top-k 3 --failed-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from prompt_siren.attack_chain_attribution import (
    attribute_attack_chain_safely,
    render_attribution_markdown,
)
from prompt_siren.attack_chain_open_coding import find_chain_sources
from prompt_siren.providers import infer_model
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

AttributionStatus = Literal["created", "skipped", "would_create", "failed"]


def create_attribution_model(model: str) -> Model:
    """Explicitly read OpenAI credentials; leave other providers unchanged."""
    prefix, separator, name = model.partition(":")
    if not separator and model.startswith("gpt-"):
        prefix, name = "openai", model
    if prefix not in {"openai", "openai-chat", "openai-responses"}:
        return infer_model(model)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing or empty. Export it in the same terminal "
            "before running this script, or use uv run --env-file .env."
        )
    provider = OpenAIProvider(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL", "").strip() or None,
    )
    model_class = OpenAIResponsesModel if prefix == "openai-responses" else OpenAIChatModel
    return model_class(name, provider=provider)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def default_benign_task(execution: dict[str, Any]) -> str:
    """Best-effort benign-task text: the first user-prompt part in the trace."""
    for message in execution.get("messages", []):
        if not isinstance(message, dict) or message.get("kind") != "request":
            continue
        for part in message.get("parts", []):
            if isinstance(part, dict) and part.get("part_kind") == "user-prompt":
                content = part.get("content")
                if isinstance(content, str) and content.strip():
                    return content
    return str(execution.get("task_id") or "")


def default_attacker_objective(payload: dict[str, Any]) -> str:
    """Best-effort attacker-objective text: the extracted chain's attack payload context."""
    contexts = payload.get("attack_context") or []
    texts = [
        item["content"]
        for item in contexts
        if isinstance(item, dict)
        and isinstance(item.get("content"), str)
        and item["content"].strip()
    ]
    return "\n\n---\n\n".join(texts)


def attack_already_succeeded_per_outcome(execution_path: Path) -> bool | None:
    """Read the attack outcome from result.json; never inferred from the attribution prompt."""
    result_path = execution_path.with_name("result.json")
    if not result_path.exists():
        return None
    attack_score = _load_json(result_path).get("attack_score")
    return None if attack_score is None else bool(attack_score)


async def attribute_file(
    chain_path: Path,
    *,
    model: str,
    top_k: int | None,
    max_attempts: int,
    max_output_tokens: int,
    benign_task: str | None,
    attacker_objective: str | None,
    failed_only: bool,
    overwrite: bool,
    dry_run: bool,
    resolved_model: Model | None = None,
) -> tuple[AttributionStatus, str]:
    json_path = chain_path.with_name("attack_chain_attribution.json")
    md_path = chain_path.with_name("attack_chain_attribution.md")
    if not overwrite and (json_path.exists() or md_path.exists()):
        return "skipped", "attribution output already exists"

    execution_path = chain_path.with_name("execution.json")
    if failed_only:
        already_succeeded = attack_already_succeeded_per_outcome(execution_path)
        if already_succeeded is None or already_succeeded:
            return "skipped", "--failed-only excludes chains without a confirmed failed outcome"
    if dry_run:
        return "would_create", str(json_path)

    payload = _load_json(chain_path)
    execution = _load_json(execution_path)
    messages = execution.get("messages")
    if not isinstance(messages, list):
        raise ValueError("execution.json does not contain a messages list")

    trajectory_id = str(payload.get("run_id") or execution.get("run_id") or chain_path.parent.name)
    resolved_benign_task = (
        benign_task if benign_task is not None else default_benign_task(execution)
    )
    resolved_attacker_objective = (
        attacker_objective
        if attacker_objective is not None
        else default_attacker_objective(payload)
    )

    settings: dict[str, Any] = {"max_tokens": max_output_tokens}
    if "qwen" in model.casefold():
        settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

    result = await attribute_attack_chain_safely(
        payload,
        messages,
        trajectory_id=trajectory_id,
        benign_task=resolved_benign_task,
        attacker_objective=resolved_attacker_objective,
        model=resolved_model if resolved_model is not None else create_attribution_model(model),
        model_settings=settings,
        top_k=top_k,
        max_attempts=max_attempts,
    )
    outputs = {
        json_path: json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        md_path: render_attribution_markdown(result),
    }
    for path, text in outputs.items():
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    status: AttributionStatus = "failed" if result["status"] == "failed" else "created"
    return (
        status,
        f"status={result['status']}, selected_message_id={result.get('selected_message_id')}",
    )


def attribution_summary_row(chain_path: Path) -> dict[str, Any]:
    """Read one trajectory's current on-disk attribution output, if any."""
    attribution_path = chain_path.with_name("attack_chain_attribution.json")
    row: dict[str, Any] = {"run_dir": str(chain_path.parent)}
    if not attribution_path.is_file():
        return row | {"status": "not_attributed", "candidates": 0, "selected_message_id": None}
    try:
        data = _load_json(attribution_path)
    except (OSError, ValueError):
        return row | {"status": "unreadable", "candidates": 0, "selected_message_id": None}
    return row | {
        "trajectory_id": data.get("trajectory_id"),
        "status": data.get("status"),
        "candidates": len(data.get("ranked_candidates") or []),
        "selected_message_id": data.get("selected_message_id"),
    }


def render_attribution_summary_markdown(rows: list[dict[str, Any]]) -> str:
    """Render a short index of which trajectories have ranked replay candidates."""
    counts = Counter(str(row["status"]) for row in rows)
    lines = ["# Attack-chain attribution summary", "", f"- Trajectories scanned: {len(rows)}"]
    lines.extend(f"  - {status}: {count}" for status, count in sorted(counts.items()))

    with_candidates = [row for row in rows if row["candidates"] > 0]
    lines.extend(["", f"## Trajectories with ranked candidates ({len(with_candidates)})", ""])
    if with_candidates:
        lines.extend(
            ["| Run | Status | Candidates | Selected message |", "| --- | --- | --- | --- |"]
        )
        lines.extend(
            f"| {row['run_dir']} | {row['status']} | {row['candidates']} | "
            f"{row['selected_message_id']} |"
            for row in with_candidates
        )
    else:
        lines.append("_None._")

    without_candidates = [row for row in rows if row["candidates"] == 0]
    lines.extend(["", f"## Trajectories without ranked candidates ({len(without_candidates)})", ""])
    if without_candidates:
        lines.extend(["| Run | Status |", "| --- | --- |"])
        lines.extend(f"| {row['run_dir']} | {row['status']} |" for row in without_candidates)
    else:
        lines.append("_None._")

    return "\n".join(lines).rstrip() + "\n"


def default_summary_path(inputs: list[Path]) -> Path:
    """Place the summary at the common root of all scanned inputs."""
    resolved_roots = [(path if path.is_dir() else path.parent).resolve() for path in inputs]
    common = (
        resolved_roots[0]
        if len(resolved_roots) == 1
        else Path(os.path.commonpath([str(root) for root in resolved_roots]))
    )
    return common / "attack_chain_attribution_summary.md"


def write_attribution_summary(chain_paths: list[Path], summary_path: Path) -> None:
    rows = [attribution_summary_row(chain_path) for chain_path in chain_paths]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_attribution_summary_markdown(rows), encoding="utf-8")


async def async_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--benign-task", default=None, help="Override the auto-derived benign-task text."
    )
    parser.add_argument(
        "--attacker-objective",
        default=None,
        help="Override the auto-derived attacker-objective text.",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Only attribute chains whose result.json records a failed attack (outcome metadata).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help=(
            "Where to write the attribution summary Markdown. Defaults to "
            "attack_chain_attribution_summary.md at the common root of the inputs."
        ),
    )
    args = parser.parse_args()
    sources = find_chain_sources(args.inputs)
    if not sources:
        raise SystemExit("No attack chains with adjacent execution.json found")
    resolved_model = None
    if not args.dry_run:
        try:
            resolved_model = create_attribution_model(args.model)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    failed = 0
    for source in sources:
        try:
            status, detail = await attribute_file(
                source.chain_path,
                model=args.model,
                top_k=args.top_k,
                max_attempts=max(1, args.max_attempts),
                max_output_tokens=max(1, args.max_output_tokens),
                benign_task=args.benign_task,
                attacker_objective=args.attacker_objective,
                failed_only=args.failed_only,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                resolved_model=resolved_model,
            )
        except Exception as exc:
            status, detail = "failed", f"{type(exc).__name__}: {exc}"
        failed += status == "failed"
        print(f"{status}: {source.chain_path} ({detail})")

    summary_path = args.summary_path or default_summary_path(args.inputs)
    write_attribution_summary([source.chain_path for source in sources], summary_path)
    print(f"summary: {summary_path}")
    if failed:
        raise SystemExit(1)


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
