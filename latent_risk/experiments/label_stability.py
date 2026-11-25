"""Stability of the replay labels under the screening rule (driver: label_stability.sh). Reads only.

Screening rule (census_label.sh, LabelRule.screen): 5 replays; at or below the pair's fresh-run
rate is safe; otherwise 5 more and the label on all 10. For every screening candidate (a label
with at least 10 replays whose first 5 were above the pair's baseline rate), this splits its
replays into the first 5 and replays 6-10, ordered by when each replay's result.json was written
(replays run concurrently, so the order is approximate), and reports whether replays 6-10 alone are
still above the baseline rate, whether they alone are significant (one-sided Fisher, as the
label), the mean rate in each half, and the time between replays 5 and 6 (a long gap makes the
second half closer to a re-test). Dangerous labels are split by whether the message is the offline
first deviation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import end_to_end as e2e
from claim2_audit import first_chain_message
from scipy.stats import fisher_exact
from scripts.task_deviation_attribution import write_atomic


@dataclass
class Row:
    first: int  # attack successes in replays 1-5
    extra: int  # attack successes in replays 6-10
    baseline: tuple[int, int]  # the pair's fresh runs: attack successes, runs
    gap_hours: float  # between the 5th and the 6th result

    @property
    def rate(self) -> float:
        return self.baseline[0] / self.baseline[1]

    def above(self, k: int) -> bool:
        bk, bn = self.baseline
        return k * bn > bk * 5

    @property
    def extra_significant(self) -> bool:
        bk, bn = self.baseline
        table = [[self.extra, 5 - self.extra], [bk, bn - bk]]
        return float(fisher_exact(table, alternative="greater")[1]) < 0.05


def ordered_results(cut_dirs: list[str]) -> list[tuple[float, bool]]:
    """(time, attack success) of every finished replay of the cuts, oldest result first."""
    results = [
        result
        for cut in cut_dirs
        for result in Path(cut).glob("*/*/result.json")
        if "__truncated_" not in result.parent.name
    ]
    return sorted(
        (path.stat().st_mtime, bool(json.loads(path.read_text()).get("attack_score")))
        for path in results
    )


def line(group: str, rows: list[Row]) -> str:
    n = len(rows)
    first = sum(r.first for r in rows) / (5 * n)
    extra = sum(r.extra for r in rows) / (5 * n)
    gaps = [r.gap_hours for r in rows]
    return (
        f"| {group} | {n} | {sum(r.above(r.extra) for r in rows)}/{n} | "
        f"{sum(r.extra_significant for r in rows)}/{n} | {first:.0%} | {extra:.0%} | "
        f"{sum(r.rate for r in rows) / n:.0%} | {statistics.median(gaps):.1f} "
        f"({sum(g >= 24 for g in gaps)}/{n} over a day) |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("rq3/dataset_v2/manifest_full.json"))
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["gpt:skill_only", "gpt:standard", "qwen:skill_only", "qwen:standard"],
    )
    parser.add_argument("--out", type=Path, default=Path("truncated/label_stability/report.md"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cell_args = argparse.Namespace(
        manifest=args.manifest,
        screen=True,
        out=args.out.parent,
        state=args.out.parent,
        replay_concurrency=1,
    )
    groups: dict[str, list[Row]] = {}
    for name in args.cells:
        cell = e2e.Cell(name, cell_args)
        sources = cell.study.sources
        for (key, index), label in cell.study.labels().items():
            if label.verdict not in ("dangerous", "safe"):
                continue
            results = ordered_results(label.cut_dirs)
            if len(results) < 10:
                continue
            row = Row(
                first=sum(hit for _, hit in results[:5]),
                extra=sum(hit for _, hit in results[5:10]),
                baseline=label.baseline,
                gap_hours=(results[5][0] - results[4][0]) / 3600,
            )
            if not row.above(row.first):
                continue  # not a screening candidate (e.g. an older label replayed 10+ times)
            at_fd = bool(sources.chain(key)) and first_chain_message(sources, key) == index
            kind = (
                "candidate, safe on 10"
                if label.verdict == "safe"
                else "dangerous, at the first deviation"
                if at_fd
                else "dangerous, elsewhere"
            )
            for group in (kind, f"{kind} | {name}"):
                groups.setdefault(group, []).append(row)
    lines = [
        "# Stability of the screening labels: first 5 replays vs replays 6-10",
        "",
        f"Cells {', '.join(args.cells)} (`{args.manifest}`). Screening candidates: labels with at "
        "least 10 replays whose first 5 were above the task pair's fresh-run rate (census first "
        "deviations and runtime firing points). Replays ordered by result time (approximate: "
        "replays run concurrently). *Above baseline* = rate of replays 6-10 above the fresh-run "
        "rate; *significant* = one-sided Fisher test vs the fresh runs, p < 0.05, on replays 6-10 "
        "alone (hard to reach with 5 replays unless the baseline is low). Gap = hours between the "
        "5th and the 6th result.",
        "",
        "| Labels | n | Replays 6-10 above baseline | Replays 6-10 alone significant | "
        "ASR, first 5 | ASR, 6-10 | Mean baseline | Gap 5th to 6th, median hours |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
        *(line(g, groups[g]) for g in sorted(groups, key=lambda g: (" | " in g, g))),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out, "\n".join(lines) + "\n")
    print(args.out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
