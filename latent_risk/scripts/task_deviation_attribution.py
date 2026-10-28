#!/usr/bin/env python3
"""Payload-blind workflow: task-deviation chain extraction, then attribution.

For every run under the inputs, the extractor sees only the user's task and the trajectory, and
selects the messages that belong to work the task does not explain
(``latent_risk.task_deviation``). The attribution judge then ranks that chain's assistant
messages, again without the attacker objective or payload. Neither judge is told whether the run
was attacked.

Outputs never go into the job directories: for an input directory D, each run's files are written
to <output-dir>/<D's name>/<run path under D>/:
  task_deviation_chain.json / .md         the extracted chain (an AttackChainJudgeAnalysis)
  task_deviation_attribution.json / .md   ranked candidates (same format as attack-chain attribution)

Example:
    uv run --env-file .env python latent_risk/scripts/task_deviation_attribution.py \\
        jobs/groups --output-dir rq3/task_deviation --model gpt-5.6-luna --top-k 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

LATENT_RISK_ROOT = Path(__file__).resolve().parents[1]
if str(LATENT_RISK_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_RISK_ROOT))

from latent_risk.attack_chain_attribution import (  # noqa: E402
    attribute_attack_chain_safely,
    render_attribution_markdown,
    with_preceding_context,
)
from latent_risk.attack_chain_judge import render_attack_chain_markdown  # noqa: E402
from latent_risk.task_deviation import extract_task_deviation_chain  # noqa: E402
from scripts.attribute_attack_chains import (  # noqa: E402
    create_attribution_model,
    default_benign_task,
)

CHAIN = "task_deviation_chain.json"
ATTRIBUTION = "task_deviation_attribution.json"


def find_runs(root: Path) -> list[Path]:
    """Original runs only: no truncated checkpoints or parked replays."""
    if (root / "execution.json").exists():
        return [root]
    return sorted(
        path.parent
        for path in root.rglob("execution.json")
        if "__truncated_" not in str(path) and "_interrupted" not in path.parts
    )


def output_dir_for(run_dir: Path, root: Path, output_root: Path) -> Path:
    if run_dir == root:
        return output_root / run_dir.name
    return output_root / root.name / run_dir.relative_to(root)


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def model_settings(model: str, max_output_tokens: int) -> ModelSettings:
    settings = ModelSettings(max_tokens=max_output_tokens)
    if "qwen" in model.casefold():
        settings["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    return settings


async def process_run(
    run_dir: Path,
    out_dir: Path,
    *,
    args: argparse.Namespace,
    extraction_model: Model,
    attribution_model: Model,
) -> str:
    execution = json.loads((run_dir / "execution.json").read_text(encoding="utf-8"))
    messages = execution.get("messages")
    if not isinstance(messages, list):
        return "skipped: execution.json has no messages list"
    benign_task = default_benign_task(execution)
    chain_path = out_dir / CHAIN
    if chain_path.exists() and not args.overwrite:
        chain = json.loads(chain_path.read_text(encoding="utf-8"))
    else:
        analysis = await extract_task_deviation_chain(
            messages,
            benign_task=benign_task,
            model=extraction_model,
            model_settings=model_settings(args.model, args.max_output_tokens),
            max_attempts=args.max_attempts,
        )
        chain = {"run_id": run_dir.name, **analysis.to_json()}
        write_atomic(chain_path, json.dumps(chain, ensure_ascii=False, indent=2) + "\n")
        write_atomic(chain_path.with_suffix(".md"), render_attack_chain_markdown(chain, messages))
    attribution_path = out_dir / ATTRIBUTION
    if args.extract_only or (attribution_path.exists() and not args.overwrite):
        return f"chain={len(chain.get('attack_message_indices', []))} messages"
    result = await attribute_attack_chain_safely(
        with_preceding_context(chain, messages, args.context_before, args.context_after),
        messages,
        trajectory_id=run_dir.name,
        benign_task=benign_task,
        attacker_objective=None,
        model=attribution_model,
        model_settings=model_settings(args.attribution_model, args.max_output_tokens),
        top_k=args.top_k,
        max_attempts=args.max_attempts,
        require_scores=args.scores,
    )
    write_atomic(attribution_path, json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    write_atomic(attribution_path.with_suffix(".md"), render_attribution_markdown(result))
    return (
        f"chain={len(chain.get('attack_message_indices', []))} messages, "
        f"status={result['status']}, top={result.get('selected_message_id')}"
    )


async def async_main(args: argparse.Namespace) -> int:
    plans = [
        (run_dir, output_dir_for(run_dir, root, args.output_dir))
        for root in args.inputs
        for run_dir in find_runs(root)
    ]
    if args.run_ids:
        wanted = set(args.run_ids)
        plans = [plan for plan in plans if plan[0].name in wanted]
    print(f"{len(plans)} run(s)", flush=True)
    if args.dry_run:
        for run_dir, out_dir in plans:
            print(f"{run_dir} -> {out_dir}")
        return 0
    extraction_model = create_attribution_model(args.model)
    attribution_model = (
        extraction_model
        if args.attribution_model == args.model
        else create_attribution_model(args.attribution_model)
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    failures = 0

    async def one(run_dir: Path, out_dir: Path) -> None:
        nonlocal failures
        async with semaphore:
            try:
                status = await process_run(
                    run_dir,
                    out_dir,
                    args=args,
                    extraction_model=extraction_model,
                    attribution_model=attribution_model,
                )
            except Exception as exc:
                failures += 1
                status = f"failed: {type(exc).__name__}: {exc}"
            print(f"{run_dir}: {status}", flush=True)

    await asyncio.gather(*(one(run_dir, out_dir) for run_dir, out_dir in plans))
    return 1 if failures else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="job directories or run directories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="extraction model")
    parser.add_argument(
        "--attribution-model", default=None, help="attribution model (default: --model)"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--context-before",
        type=int,
        default=0,
        help="add up to N preceding non-assistant messages of each chain assistant message",
    )
    parser.add_argument(
        "--context-after",
        type=int,
        default=0,
        help="add up to N following non-assistant messages (e.g. 1: the next tool output)",
    )
    parser.add_argument("--scores", action="store_true", help="ask for a 0-100 risk_score")
    parser.add_argument("--run-ids", nargs="+", default=None, help="only these run ids")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    args.attribution_model = args.attribution_model or args.model
    return args


def main() -> None:
    raise SystemExit(asyncio.run(async_main(parse_args())))


if __name__ == "__main__":
    main()
