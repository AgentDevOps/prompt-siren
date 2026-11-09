"""Effect of the inferred warning at flagged firings (driver: warning_effect.sh). Reads only.

From the end-to-end test (end_to_end.py: every flagged firing replayed without the warning, the
control, and with it, at the same time): per group of flagged firings, the task pairs' baseline
ASR (fresh runs, averaged over the firings), the replayed ASR without and with the warning, the
drop, and the benign task score in both arms. Groups: attack-succeeded runs (cut after the firing,
before the attack action) and attack-failed runs, split by the replay label of the firing point
(prevalence labels, any replays in truncated/; the control arm of the end-to-end test is under
truncated/inserted_context and never counts as a label).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import end_to_end as e2e
import recovery_context as rc
from scripts.task_deviation_attribution import write_atomic

CELLS = ("gpt:skill_only", "gpt:standard")
GROUPS = (
    "all flagged firings",
    "attack-succeeded runs",
    "attack-failed runs",
    "attack-failed, firing labelled dangerous",
    "attack-failed, firing labelled safe",
    "attack-failed, firing uncertain or unlabelled",
)


@dataclass
class Group:
    firings: int = 0
    pairs: set[str] = field(default_factory=set)
    baseline: float = 0.0
    control: list[int] = field(default_factory=lambda: [0, 0])  # replays, attack successes
    warned: list[int] = field(default_factory=lambda: [0, 0])
    benign: list[float] = field(default_factory=lambda: [0.0, 0.0])  # summed: control, warned

    def add(
        self,
        pair: str,
        base: float,
        control: tuple[int, int, float],
        warned: tuple[int, int, float],
    ) -> None:
        self.firings += 1
        self.pairs.add(pair)
        self.baseline += base
        self.control[0] += control[0]
        self.control[1] += control[1]
        self.warned[0] += warned[0]
        self.warned[1] += warned[1]
        self.benign[0] += control[2]
        self.benign[1] += warned[2]

    def merge(self, other: Group, cell: str) -> None:
        self.firings += other.firings
        self.pairs |= {f"{cell}/{pair}" for pair in other.pairs}
        self.baseline += other.baseline
        for mine, theirs in ((self.control, other.control), (self.warned, other.warned)):
            mine[0] += theirs[0]
            mine[1] += theirs[1]
        self.benign[0] += other.benign[0]
        self.benign[1] += other.benign[1]

    def row(self, name: str) -> str:
        if not self.firings or not self.control[0] or not self.warned[0]:
            return f"| {name} | {self.firings} | | | | | | | |"
        none = self.control[1] / self.control[0]
        warn = self.warned[1] / self.warned[0]
        relative = f"{(none - warn) / none:.0%}" if none else "-"
        return (
            f"| {name} | {self.firings} | {len(self.pairs)} | {self.baseline / self.firings:.0%} | "
            f"{self.control[1]}/{self.control[0]} = {none:.0%} | "
            f"{self.warned[1]}/{self.warned[0]} = {warn:.0%} | "
            f"{100 * (none - warn):.0f} points ({relative}) | "
            f"{self.benign[0] / self.control[0]:.2f} | {self.benign[1] / self.warned[0]:.2f} |"
        )


def groups_for(name: str, args: argparse.Namespace) -> dict[str, Group]:
    cell = e2e.Cell(name, args)
    decisions = json.loads((cell.out(args) / "decisions.json").read_text())
    labels = cell.study.labels()
    state = cell.state(args)
    groups = {g: Group() for g in GROUPS}
    for key, entry in decisions.items():
        if not entry.get("flagged") or key not in cell.outcome:
            continue
        item = (key, entry["step"])
        control = e2e.arm(rc.cut_dir(cell.study, state, item, e2e.CONTROL))
        warned = e2e.arm(rc.cut_dir(cell.study, state, item, e2e.WARNED))
        if not control[0] or not warned[0]:
            continue
        job = cell.study.sources.job_of[key]
        values = (cell.study.pair(key), job.base_k / job.base_n, control, warned)
        names = ["all flagged firings"]
        if cell.outcome[key][0]:
            names.append("attack-succeeded runs")
        else:
            label = labels.get(item)
            verdict = label.verdict if label is not None else "unlabelled"
            names.append("attack-failed runs")
            names.append(
                {
                    "dangerous": "attack-failed, firing labelled dangerous",
                    "safe": "attack-failed, firing labelled safe",
                }.get(verdict, "attack-failed, firing uncertain or unlabelled")
            )
        for group in names:
            groups[group].add(*values)
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("rq3/dataset_v2/manifest.json"))
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--e2e-out", type=Path, default=Path("truncated/end_to_end"))
    parser.add_argument("--state", type=Path, default=Path("truncated/inserted_context/end_to_end"))
    parser.add_argument("--out", type=Path, default=Path("truncated/warning_effect/report.md"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    e2e_args = argparse.Namespace(
        manifest=args.manifest, out=args.e2e_out, state=args.state, replay_concurrency=1
    )
    header = [
        "| Flagged firings | Firings | Task pairs | Baseline ASR (fresh runs) | Replayed ASR, "
        "no warning | Replayed ASR, warning | Drop (relative) | Benign score, no warning | "
        "Benign score, warning |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines = [
        "# Effect of the inferred warning at flagged firings",
        "",
        f"End-to-end test `{args.e2e_out}` (manifest `{args.manifest}`). Each flagged firing is "
        "replayed up to 5 times without and with the warning, at the same time. Baseline = the "
        "task pair's fresh-run ASR, averaged over the firings. Benign score = share of the task's "
        "tests passed, mean over replays. Labels: replay labels of the firing point "
        "(attack-failed runs only).",
    ]
    total = {g: Group() for g in GROUPS}
    for name in args.cells:
        groups = groups_for(name, e2e_args)
        lines += ["", f"## {name}", "", *header, *(groups[g].row(g) for g in GROUPS)]
        for g in GROUPS:
            total[g].merge(groups[g], name)
    lines += ["", "## all cells", "", *header, *(total[g].row(g) for g in GROUPS)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out, "\n".join(lines) + "\n")
    print(args.out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
