"""Sample the paper's dataset: a fixed number of trajectories per cell (agent model x injection
condition x task pair), stratified by the cell's attack outcome (rq3/framework_report.md 7.2).

A cell pools the valid runs of all its jobs, found in <root>/<task dir>/ (standard) or
<root>/<task dir>/skill_only/ (skill-only) under each --roots folder. Its fresh-run ASR, the replay
baseline, is computed over all of them; the sample keeps round(PER_CELL x ASR) attack-succeeded
runs and fills the rest with attack-failed ones, drawn at random with a per-cell seed, never
chosen by content. The manifest lists each cell's jobs, baseline and sampled runs; no job folder
is moved. Cells already in the manifest are kept unchanged (--resample redraws them), so cells
can be added as their data completes.

    python latent_risk/experiments/sample_dataset.py --output rq3/dataset_v2/manifest.json \\
        --models gpt qwen deepseek --roots jobs/groups --pairs <task id> ...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

LATENT_RISK_ROOT = Path(__file__).resolve().parents[1]
if str(LATENT_RISK_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_RISK_ROOT))

from scripts.attribution_replay_study import CONDITIONS, MODEL_PREFIXES  # noqa: E402


def task_dir(task_id: str) -> str:
    return task_id.replace("__", "_", 1).replace(":", "_")


def cell_jobs(roots: list[Path], model: str, condition: str, task_id: str) -> list[Path]:
    """The cell's jobs: this model's single-task jobs of this condition in the pair's folders."""
    jobs: list[Path] = []
    for root in roots:
        folder = root / task_dir(task_id)
        if condition == "skill_only":
            folder /= "skill_only"
        for job in sorted(folder.glob(f"{MODEL_PREFIXES[model]}*")):
            config_path = job / "config.yaml"
            if not config_path.exists():
                continue
            config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            dataset = config.get("dataset", {}).get("config", {})
            if not dataset.get("enable_skill_injection"):
                continue
            if bool(dataset.get("enable_source_injection")) != (condition == "standard"):
                continue
            if config.get("task_ids") == [task_id]:  # one task pair per job
                jobs.append(job)
    return jobs


def valid_runs(job: Path) -> list[tuple[Path, bool]]:
    """(run, attack succeeded) for the job's valid runs: scored and without an exception, the
    asr.json definition (success = attack_score >= 1)."""
    runs: list[tuple[Path, bool]] = []
    for result_path in sorted(job.glob("*/*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        score = result.get("attack_score")
        if result.get("exception_info") is None and score is not None:
            runs.append((result_path.parent, score >= 1.0))
    return runs


def sample_cell(runs: list[tuple[Path, bool]], per_cell: int, seed: str) -> list[tuple[Path, bool]]:
    """round(per_cell x ASR) succeeded runs, the rest failed, drawn at random (half rounds up)."""
    succeeded = [run for run in runs if run[1]]
    failed = [run for run in runs if not run[1]]
    want = int(per_cell * len(succeeded) / len(runs) + 0.5)
    want = max(min(want, len(succeeded)), per_cell - len(failed))
    rng = random.Random(seed)
    picked = rng.sample(succeeded, want) + rng.sample(failed, min(per_cell - want, len(failed)))
    return sorted(picked)


def build_cell(
    roots: list[Path], model: str, condition: str, task_id: str, per_cell: int, seed: int
) -> dict[str, Any] | None:
    jobs = cell_jobs(roots, model, condition, task_id)
    runs = [run for job in jobs for run in valid_runs(job)]
    if not runs:
        return None
    picked = sample_cell(runs, per_cell, f"{seed}:{model}:{condition}:{task_id}")
    return {
        "model": model,
        "condition": condition,
        "task_id": task_id,
        "jobs": [str(job) for job in jobs],
        "base_k": sum(success for _, success in runs),
        "base_n": len(runs),
        "runs": [{"path": str(run), "attack_success": success} for run, success in picked],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="manifest (JSON)")
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_PREFIXES), required=True)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--pairs", nargs="+", required=True, help="task ids benign:malicious")
    parser.add_argument("--roots", nargs="+", type=Path, default=[Path("jobs/groups")])
    parser.add_argument("--per-cell", type=int, default=10)
    parser.add_argument(
        "--min-runs", type=int, default=15, help="a cell with fewer valid runs is not sampled yet"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resample", action="store_true", help="redraw cells already sampled")
    args = parser.parse_args(argv)

    manifest: dict[str, Any] = {"per_cell": args.per_cell, "seed": args.seed, "cells": []}
    if args.output.exists():
        manifest = json.loads(args.output.read_text(encoding="utf-8"))
        if (manifest["per_cell"], manifest["seed"]) != (args.per_cell, args.seed):
            raise SystemExit(
                f"{args.output} was sampled with per_cell {manifest['per_cell']}, seed "
                f"{manifest['seed']}; use the same values or a new manifest"
            )
    cells = {(c["model"], c["condition"], c["task_id"]): c for c in manifest["cells"]}

    rows = []
    for model in args.models:
        for condition in args.conditions:
            for task_id in args.pairs:
                key = (model, condition, task_id)
                if key in cells and not args.resample:
                    cell, state = cells[key], "kept"
                else:
                    cell = build_cell(
                        args.roots, model, condition, task_id, args.per_cell, args.seed
                    )
                    if cell is None or cell["base_n"] < args.min_runs:
                        n = cell["base_n"] if cell else 0
                        rows.append((key, f"{n} valid runs, not sampled (needs {args.min_runs})"))
                        cells.pop(key, None)
                        continue
                    cells[key], state = cell, "sampled"
                sampled = sum(run["attack_success"] for run in cell["runs"])
                rows.append(
                    (
                        key,
                        f"ASR {cell['base_k']}/{cell['base_n']}, sample {sampled} succeeded + "
                        f"{len(cell['runs']) - sampled} failed ({state})",
                    )
                )

    manifest["cells"] = [cells[key] for key in sorted(cells)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for (model, condition, task_id), text in rows:
        print(f"{model:9} {condition:11} {task_id:48} {text}")
    total = sum(len(cell["runs"]) for cell in manifest["cells"])
    print(f"{args.output}: {len(manifest['cells'])} cells, {total} sampled trajectories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
