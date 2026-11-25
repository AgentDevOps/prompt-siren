"""Census manifest of dataset v2 (driver: full_manifest.sh): every valid run of each cell.

Reads a sampled manifest (sample_dataset.py) and writes a manifest with the same cells, jobs and
baselines but all valid runs of each cell's jobs instead of the sample (sample_dataset.valid_runs:
scored and without an exception). Each cell's base_k/base_n is checked against the runs. The input
manifest is unchanged. Prints, per model x condition, how many attack-failed runs the census adds.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from sample_dataset import valid_runs
from scripts.task_deviation_attribution import write_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("rq3/dataset_v2/manifest_swap.json"))
    parser.add_argument("--output", type=Path, default=Path("rq3/dataset_v2/manifest_full.json"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("--output must differ from --input")
    manifest = json.loads(args.input.read_text())
    added: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    cells = []
    for cell in manifest["cells"]:
        runs = [run for job in cell["jobs"] for run in valid_runs(Path(job))]
        successes = sum(won for _, won in runs)
        if (successes, len(runs)) != (cell["base_k"], cell["base_n"]):
            print(
                f"warning: {cell['model']} {cell['condition']} {cell['task_id']}: runs "
                f"{successes}/{len(runs)} vs baseline {cell['base_k']}/{cell['base_n']}"
            )
        sampled = {Path(run["path"]).name for run in cell["runs"]}
        failed = [path for path, won in runs if not won]
        name = f"{cell['model']} {cell['condition']}"
        added[name][0] += len(failed)
        added[name][1] += sum(path.name not in sampled for path in failed)
        added[name][2] += 1
        cells.append(
            cell | {"runs": [{"path": str(path), "attack_success": won} for path, won in runs]}
        )
    output = manifest | {"per_cell": "all", "source": str(args.input), "cells": cells}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.output, json.dumps(output, indent=2) + "\n")
    for name, (failed, new, pairs) in sorted(added.items()):
        print(f"{name}: {pairs} task pairs, {failed} attack-failed runs ({new} not in the sample)")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
