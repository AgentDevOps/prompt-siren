"""Dangerous messages in dataset v2, per cell (driver: v2_dangerous.sh).

For each model x condition x task pair of the manifest: attack-failed runs in the sample, labelled
messages, dangerous messages (any message, and first deviations), trajectories holding one, the
pair's baseline ASR (all fresh runs) and the replayed ASR (pooled replays over its dangerous
messages, and the range per message). Labels as in prevalence_baselines.py. Reads only.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import fd_risk_score as frs
from claim2_audit import first_chain_message
from scripts.task_deviation_attribution import write_atomic

CELLS = ("gpt:skill_only", "gpt:standard", "qwen:standard", "qwen:skill_only")


@dataclass
class Row:
    base: tuple[int, int] = (0, 0)
    failed: int = 0
    labelled: int = 0
    dangerous: int = 0
    first: int = 0
    before: int = 0
    deviations: int = 0
    deviations_safe: int = 0
    successes: int = 0
    replays: int = 0
    runs: set[str] = field(default_factory=set)
    rates: list[float] = field(default_factory=list)
    matched: list[float] = field(default_factory=list)

    def add(self, other: Row) -> None:
        self.base = (self.base[0] + other.base[0], self.base[1] + other.base[1])
        self.failed += other.failed
        self.labelled += other.labelled
        self.dangerous += other.dangerous
        self.first += other.first
        self.before += other.before
        self.deviations += other.deviations
        self.deviations_safe += other.deviations_safe
        self.successes += other.successes
        self.replays += other.replays
        self.runs |= other.runs


def replayed(row: Row, with_range: bool) -> str:
    if not row.replays:
        return "-"
    text = f"{row.successes}/{row.replays} = {row.successes / row.replays:.0%}"
    if with_range and len(row.rates) > 1:
        text += f" ({min(row.rates):.0%} to {max(row.rates):.0%})"
    return text


def baseline(row: Row) -> str:
    k, n = row.base
    return f"{k}/{n} = {k / n:.0%}" if n else "-"


def matched(row: Row) -> str:
    return f"{sum(row.matched) / len(row.matched):.0%}" if row.matched else "-"


def positions(row: Row) -> str:
    later = row.dangerous - row.first - row.before
    return f"{row.before} / {row.first} / {later}"


def split(row: Row) -> str:
    other = row.deviations - row.first - row.deviations_safe
    return f"{row.first} / {row.deviations_safe} / {other}"


def cell_rows(name: str, manifest: Path) -> dict[str, Row]:
    study = frs.load_study(name, argparse.Namespace(manifest=manifest))
    rows: dict[str, Row] = {}
    for key in study.sources.runs:
        job = study.sources.job_of[key]
        row = rows.setdefault(study.pair(key), Row(base=(job.base_k, job.base_n)))
        row.failed += 1
    labels = study.labels()
    for key in study.sources.runs:
        index = first_chain_message(study.sources, key) if study.sources.chain(key) else None
        if index is None:
            continue
        row = rows[study.pair(key)]
        row.deviations += 1
        label = labels.get((key, index))
        row.deviations_safe += label is not None and label.verdict == "safe"
    for (key, index), label in labels.items():
        if key not in study.sources.runs or label.verdict not in ("dangerous", "safe"):
            continue
        row = rows[study.pair(key)]
        row.labelled += 1
        if label.verdict == "dangerous":
            row.dangerous += 1
            first = first_chain_message(study.sources, key) if study.sources.chain(key) else None
            row.first += index == first
            row.before += first is None or index < first
            row.runs.add(key)
            row.successes += label.successes
            row.replays += label.replays
            row.rates.append(label.successes / label.replays)
            row.matched.append(row.base[0] / row.base[1])
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("rq3/dataset_v2/manifest_swap.json"))
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--out", type=Path, default=Path("truncated/v2_dangerous/report.md"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    lines = [
        "# Dangerous messages in dataset v2",
        "",
        f"Manifest `{args.manifest}`. Per task pair: attack-failed runs in the sample; labelled "
        "messages (dangerous or safe); first deviations (runs with one) and how they are "
        "labelled: dangerous / safe / uncertain or unlabelled; dangerous messages before / at / after the first "
        "deviation (before: earlier in the run, or in a run without a first deviation); trajectories "
        "holding at least one dangerous message; baseline ASR over all fresh runs of the pair; "
        "replayed ASR pooled over the pair's dangerous messages (range per message).",
        "",
    ]
    cells = {name: cell_rows(name, args.manifest) for name in args.cells}
    lines += [
        "## Per cell",
        "",
        "Baseline ASR pooled over the fresh runs of the cell's task pairs (range over pairs); "
        "baseline of the dangerous messages' own pairs (mean over dangerous messages); "
        "replayed ASR pooled over the cell's dangerous messages (range per message).",
        "",
        "| Cell | Attack-failed runs | First deviations | First deviations dangerous / safe / "
        "uncertain or unlabelled | Dangerous messages before / at / after the first deviation | Trajectories "
        "with a dangerous message | Baseline ASR (range over pairs) | Baseline ASR of the "
        "dangerous messages' pairs | Replayed ASR (range) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    everything = Row()
    for name, rows in cells.items():
        total = Row()
        for row in rows.values():
            total.add(row)
            total.rates += row.rates
            total.matched += row.matched
        everything.add(total)
        everything.rates += total.rates
        everything.matched += total.matched
        pairs = [k / n for k, n in (row.base for row in rows.values()) if n]
        lines.append(
            f"| {name.replace(':', ' ')} | {total.failed} | {total.deviations} | {split(total)} | "
            f"{positions(total)} | {len(total.runs)} | "
            f"{baseline(total)} ({min(pairs):.0%} to {max(pairs):.0%}) | {matched(total)} | {replayed(total, True)} |"
        )
    lines += [
        f"| **all** | **{everything.failed}** | **{everything.deviations}** | "
        f"**{split(everything)}** | **{positions(everything)}** "
        f"| **{len(everything.runs)}** | **{baseline(everything)}** | **{matched(everything)}** | "
        f"**{replayed(everything, True)}** |",
        "",
    ]
    for name, rows in cells.items():
        lines += [
            f"## {name.replace(':', ' ')}",
            "",
            "| Task pair | Attack-failed runs | Labelled messages | First deviations | "
            "First deviations dangerous / safe / uncertain or unlabelled | Dangerous messages "
            "before / at / after the first deviation | Trajectories with a dangerous message | "
            "Baseline ASR | Replayed ASR (range) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        total = Row()
        for pair, row in sorted(rows.items()):
            k, n = row.base
            lines.append(
                f"| {pair} | {row.failed} | {row.labelled} | {row.deviations} | {split(row)} | "
                f"{positions(row)} | {len(row.runs)} | {k}/{n} = {k / n:.0%} | {replayed(row, True)} |"
            )
            total.add(row)
        lines += [
            f"| **all ({len(rows)} pairs)** | **{total.failed}** | **{total.labelled}** | "
            f"**{total.deviations}** | **{split(total)}** | "
            f"**{positions(total)}** | **{len(total.runs)}** | | "
            f"**{replayed(total, False)}** |",
            "",
        ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out, "\n".join(lines) + "\n")
    print(args.out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
