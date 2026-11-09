"""Quick check: is the replay rate a usable evolution target instead of the dangerous/safe label?

No model calls. Uses the existing fd_combined scores (variants no-samples, samples, lookahead), the
text judge's score and the replay counts of the hard first deviations (lookahead.points). The
fitness is within-pair concordance: among two points of the same task pair with different targets,
the share where the judge's score orders them the same way (ties half; 0.5 = chance). Reports:
concordance with the rate and with the label; a binomial noise ceiling (two simulated replay batches
of the same size); and, by bootstrap over task pairs, the fitness SD and how often the samples judge
beats the no-samples judge under each target.

Run from the repository root:
  PYTHONPATH=latent_risk/experiments:latent_risk .venv/bin/python latent_risk/experiments/rate_target_check.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import fd_risk_score as frs
import lookahead as la

FIELDS = ("text", "no-samples", "samples", "lookahead")
Row = dict[str, Any]


def load_rows(args: argparse.Namespace) -> list[Row]:
    variants: dict[str, dict[tuple[str, str], int]] = {}
    for variant in FIELDS[1:]:
        variants[variant] = {}
        for dataset in args.datasets:
            path = args.combined / variant / dataset.replace(":", "_") / "scores.json"
            for name, entry in json.loads(path.read_text()).items():
                variants[variant][(dataset, name)] = entry["probability"]
    rows = []
    for dataset, study, item, entry in la.points(args):
        name = frs.run_name(study, item)
        successes, total = entry["replays"]
        row: Row = {
            "pair": f"{dataset}/{entry['pair']}",
            "label": int(entry["verdict"] == "dangerous"),
            "total": total,
            "rate": successes / total,
            "text": entry["probability"],
        }
        row |= {v: scores.get((dataset, name)) for v, scores in variants.items()}
        rows.append(row)
    return rows


def concordance(
    groups: list[list[Row]], score: Callable[[Row], float | None], target: Callable[[Row], float]
) -> tuple[float | None, int]:
    agree = 0.0
    total = 0
    for group in groups:
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                sa, sb = score(a), score(b)
                if sa is None or sb is None or target(a) == target(b):
                    continue
                product = (sa - sb) * (target(a) - target(b))
                agree += 1.0 if product > 0 else 0.5 if product == 0 else 0.0
                total += 1
    return (agree / total if total else None), total


def field(name: str) -> Callable[[Row], float | None]:
    return lambda row: row[name]


def rate(row: Row) -> float:
    return row["rate"]


def label(row: Row) -> float:
    return row["label"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("rq3/dataset_v2/manifest.json"))
    parser.add_argument("--datasets", nargs="+", default=["d1", "gpt:skill_only", "gpt:standard"])
    parser.add_argument("--scores", type=Path, default=Path("truncated/fd_risk_score"))
    parser.add_argument("--combined", type=Path, default=Path("truncated/fd_combined"))
    parser.add_argument("--min-score", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    rng = random.Random(args.seed)
    rows = load_rows(args)
    by_pair: dict[str, list[Row]] = defaultdict(list)
    for row in rows:
        by_pair[row["pair"]].append(row)
    groups = list(by_pair.values())
    print(f"{len(rows)} points, {sum(r['label'] for r in rows)} dangerous, {len(groups)} pairs")
    print("Within-pair concordance (0.5 = chance): with the rate | with the label")
    for name in FIELDS:
        value, n = concordance(groups, field(name), rate)
        value_l, n_l = concordance(groups, field(name), label)
        print(f"  {name}: {value:.2f} ({n}) | {value_l:.2f} ({n_l})")

    def draw(row: Row) -> float:
        return sum(rng.random() < row["rate"] for _ in range(row["total"])) / row["total"]

    ceiling = []
    for _ in range(100):
        first = {id(r): draw(r) for r in rows}
        second = {id(r): draw(r) for r in rows}
        value, _ = concordance(groups, lambda r, s=second: s[id(r)], lambda r, f=first: f[id(r)])
        if value is not None:
            ceiling.append(value)
    print(f"Binomial noise ceiling within pair: {statistics.mean(ceiling):.2f}")
    for target_name, target in (("rate", rate), ("label", label)):
        values = []
        wins = 0
        for _ in range(args.bootstrap):
            sample = [rng.choice(groups) for _ in groups]
            better, _ = concordance(sample, field("samples"), target)
            worse, _ = concordance(sample, field("no-samples"), target)
            if better is None or worse is None:
                continue
            values.append(better)
            wins += better > worse
        print(
            f"{target_name}: samples fitness SD {statistics.stdev(values):.3f}; samples > "
            f"no-samples in {wins / len(values):.0%} of resamples"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
