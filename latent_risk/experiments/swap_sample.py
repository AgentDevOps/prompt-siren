"""Swap the most-used runs into the dataset v2 sample (driver: swap_sample.sh).

Decision (user, 2026-11-05): for the gpt standard cells of the original 8 task pairs, the v2 sample
(10 runs per pair, sample_dataset.py) is drawn from D1's runs. Runs that D1's experiments used most
(replay labels, examples, the pilot targets) are swapped into the sample, and unused sampled runs go
back to D1, keeping each pair's sample size and its number of attack-succeeded runs (so the ASR).
Only the attack-failed stratum changes: replay labels exist only on attack-failed runs.

Usage of a run = replays spent on its labelled messages, over the label folders of D1 and of the
v2 cell (the same cut counted once). Order for the attack-failed slots: --prefer run ids first,
then usage, then runs already sampled, then a seeded random order. The swap selects by usage, which
correlates with dangerous labels, so the swapped cells' prevalence is not a random-sample estimate;
the report says how many dangerous first deviations enter.

Writes a new manifest (--output); the input manifest is not changed. Commands: plan (report only),
write (report + manifest).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import fd_risk_score as frs
import prevalence_baselines as pb
from claim2_audit import first_chain_message
from sample_dataset import valid_runs
from scripts.task_deviation_attribution import write_atomic


def usage_and_danger(args: argparse.Namespace) -> tuple[dict[str, int], set[str]]:
    """Replays per run (resolved path) and the runs whose first deviation is dangerous."""
    studies = [
        pb.Study(pb.parse_args(["report"])),
        frs.load_study(f"{args.agent_model}:{args.condition}", args),
    ]
    replays: dict[tuple[str, int], int] = {}
    danger: set[str] = set()
    for study in studies:
        labels = study.labels()
        for (key, index), label in labels.items():
            path = str(study.sources.runs[key].resolve())
            replays[(path, index)] = max(replays.get((path, index), 0), label.replays)
            if label.verdict == "dangerous" and index == first_chain_message(study.sources, key):
                danger.add(path)
    usage: dict[str, int] = defaultdict(int)
    for (path, _), count in replays.items():
        usage[path] += count
    return usage, danger


def swap_cell(
    cell: dict[str, Any], usage: dict[str, int], prefer: set[str], seed: int
) -> list[dict[str, Any]]:
    runs = [run for job in cell["jobs"] for run in valid_runs(Path(job))]
    sampled = {str(Path(r["path"]).resolve()) for r in cell["runs"]}
    keep = [r for r in cell["runs"] if r["attack_success"]]
    slots = len(cell["runs"]) - len(keep)
    failed = [str(path.resolve()) for path, success in runs if not success]
    rng = random.Random(f"{seed}:{cell['task_id']}")
    noise = {path: rng.random() for path in failed}
    failed.sort(
        key=lambda p: (Path(p).name not in prefer, -usage.get(p, 0), p not in sampled, noise[p])
    )
    by_path = {str(path.resolve()): path for path, _ in runs}
    chosen = [{"path": str(by_path[p]), "attack_success": False} for p in failed[:slots]]
    return sorted(keep + chosen, key=lambda r: r["path"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "write"))
    parser.add_argument("--manifest", type=Path, default=Path("rq3/dataset_v2/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("rq3/dataset_v2/manifest_swap.json"))
    parser.add_argument("--agent-model", default="gpt")
    parser.add_argument("--condition", default="standard")
    parser.add_argument("--pairs", nargs="+", default=sorted(pb.TASK_PAIRS))
    parser.add_argument("--prefer", nargs="*", default=[], help="run ids to swap in first")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path, default=Path("truncated/swap_sample/report.md"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    manifest = json.loads(args.manifest.read_text())
    usage, danger = usage_and_danger(args)
    prefer = set(args.prefer)
    lines = [
        "# Usage-based swap of the v2 sample",
        "",
        f"Cells: {args.agent_model} {args.condition}, {len(args.pairs)} task pairs. Same sample "
        "size and attack-succeeded count per pair; attack-failed runs swapped by usage (replays "
        "on their labelled messages). Dangerous = dangerous first deviation.",
        "",
        "| Task pair | Sample (succeeded) | Swapped in | Swapped out | Replays in sample "
        "before → after | Dangerous before → after |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    totals = [0, 0, 0, 0, 0, 0]
    for cell in manifest["cells"]:
        if (cell["model"], cell["condition"]) != (args.agent_model, args.condition):
            continue
        if cell["task_id"] not in args.pairs:
            continue
        before = {str(Path(r["path"]).resolve()) for r in cell["runs"]}
        cell["runs"] = swap_cell(cell, usage, prefer, args.seed)
        after = {str(Path(r["path"]).resolve()) for r in cell["runs"]}
        row = [
            len(after - before),
            len(before - after),
            sum(usage.get(p, 0) for p in before),
            sum(usage.get(p, 0) for p in after),
            len(before & danger),
            len(after & danger),
        ]
        totals = [t + v for t, v in zip(totals, row, strict=True)]
        succeeded = sum(r["attack_success"] for r in cell["runs"])
        lines.append(
            f"| {cell['task_id']} | {len(cell['runs'])} ({succeeded}) | {row[0]} | {row[1]} | "
            f"{row[2]} → {row[3]} | {row[4]} → {row[5]} |"
        )
    lines.append(
        f"| **all** | | **{totals[0]}** | **{totals[1]}** | **{totals[2]} → {totals[3]}** | "
        f"**{totals[4]} → {totals[5]}** |"
    )
    missing = sorted(prefer - {Path(p).name for p in usage} - {Path(p).name for p in danger})
    if missing:
        lines += ["", f"Preferred run ids without labels: {', '.join(missing)}"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.report, "\n".join(lines) + "\n")
    print(args.report.read_text())
    if args.command == "write":
        if args.output.resolve() == args.manifest.resolve():
            raise SystemExit("--output must differ from --manifest")
        write_atomic(args.output, json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
