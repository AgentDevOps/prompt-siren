"""Final tables for the paper (step 11; driver: final_tables.sh). Reads only: no model calls, no
replays.

The method is R1 -> R3 -> R4 -> R5 (runtime extractor, the agent's own sampled next messages,
the next-step judge at every firing, the inferred warning); the first frozen version also had a
context-only stage 1 filter (fd-risk-v1 >= --stage1) before R3. Per cell and over all cells:

- data: task pairs, runs, attack-succeeded / attack-failed runs, fresh-run ASR;
- RQ1: the census of dangerous messages (copied from v2_dangerous.py's report) and the stability
  of the screening labels (copied from label_stability.py's report);
- RQ2: R1 against IntentGuard and AlignmentCheck (rows of alignmentcheck_baseline.py's report);
  AUC at R1's firing points in attack-failed runs (runtime_auc.py labels) for the method (R4 at
  every firing: decisions.json, plus r4_alone.py's scores where stage 1 dropped the firing), the
  context-only judge (stage 1) and the first version; flags at R4 >= --threshold; the same per
  split of held_out.py (development / held-out runs);
- RQ3: the warning effect at the method's flagged firings (warning_effect.py groups: the first
  version's flags, replays in --e2e-state, plus the flags only the method raises, replays in
  --r4-state) and end to end (end_to_end.py's estimate: each run's own outcome minus the replay
  drop at its flagged firing; bootstrap over task pairs), for the method and the first version.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import end_to_end as e2e
import held_out as ho
import recovery_context as rc
import runtime_auc as ra
import warning_effect as we
from scripts.task_deviation_attribution import write_atomic

ALL = "all cells"


def r4_scores(args: argparse.Namespace, cell: str) -> dict[str, Any]:
    path = args.r4 / cell.replace(":", "_") / "r4.json"
    return json.loads(path.read_text()) if path.exists() else {}


def copied_table(path: Path, heading: str | None, skip: list[str]) -> list[str]:
    """The first markdown table after heading (or in the file), without rows containing skip."""
    lines = path.read_text().splitlines()
    start = lines.index(heading) + 1 if heading else 0
    table: list[str] = []
    for line in lines[start:]:
        if line.startswith("|"):
            if not any(s in line for s in skip):
                table.append(line)
        elif table:
            break
    return table


def data_section(args: argparse.Namespace, cells: dict[str, e2e.Cell]) -> list[str]:
    lines = [
        "## Data",
        "",
        "| Cell | Task pairs | Runs | Attack-succeeded | Attack-failed | Fresh-run ASR "
        "(range over pairs) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    total = [0, 0, 0]
    for name, cell in cells.items():
        by_pair: dict[str, list[bool]] = {}
        for key in cell.keys:
            by_pair.setdefault(cell.study.pair(key), []).append(cell.outcome[key][0])
        runs = len(cell.keys)
        succeeded = sum(cell.outcome[k][0] for k in cell.keys)
        rates = [sum(v) / len(v) for v in by_pair.values()]
        lines.append(
            f"| {name} | {len(by_pair)} | {runs} | {succeeded} | {runs - succeeded} | "
            f"{succeeded}/{runs} = {succeeded / runs:.0%} ({min(rates):.0%} to {max(rates):.0%}) |"
        )
        total = [total[0] + len(by_pair), total[1] + runs, total[2] + succeeded]
    pairs, runs, succeeded = total
    lines.append(
        f"| **{ALL}** | {pairs} pair-condition cells | {runs} | {succeeded} | "
        f"{runs - succeeded} | {succeeded}/{runs} = {succeeded / runs:.0%} |"
    )
    return lines


def rq1_section(args: argparse.Namespace) -> list[str]:
    return [
        "## RQ1: dangerous messages in attack-failed runs (census)",
        "",
        f"From `{args.census}`: every first deviation of every attack-failed run, replay-labelled "
        "(screening rule).",
        "",
        *copied_table(args.census, "## Per cell", []),
        "",
        "### Stability of the screening labels",
        "",
        f"From `{args.stability}`: replays 1-5 vs 6-10 of every screening candidate.",
        "",
        *copied_table(args.stability, None, args.cells),
    ]


def baselines_section(args: argparse.Namespace) -> list[str]:
    rows = {
        "Attack-succeeded runs with a firing before the attack action": "AS runs fired before the attack action",
        "Attack-failed dangerous trajectories caught (at or before the dangerous message)": "AF dangerous trajectories caught",
        "Attack-failed runs without an offline first deviation that it fires in": "False alarms (AF runs never off-task)",
        "Lead in attack-succeeded runs: agent messages before the attack action, median (range)": "Lead before the attack action, messages",
    }
    lines = [
        "## RQ2: R1 against context-only detectors (all runs)",
        "",
        f"From `{args.baselines}`. Each cell: R1 / IntentGuard Plan Gate / AlignmentCheck.",
        "",
        "| Cell | " + " | ".join(rows.values()) + " |",
        "| --- | " + " | ".join("---" for _ in rows) + " |",
    ]
    text = args.baselines.read_text().splitlines()
    for name in [*args.cells, ALL]:
        start = text.index(f"## {name}")
        values = []
        for label in rows:
            line = next(row for row in text[start:] if row.startswith(f"| {label} |"))
            values.append(" / ".join(c.strip() for c in line.split("|")[2:5]))
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    return lines


def auc_cell(points: list[ra.Point], score: str, args: argparse.Namespace) -> str:
    scored = []
    for p in points:
        value = p.score(score, args.stage1)
        if value is not None and p.verdict in ("dangerous", "safe"):
            scored.append((p.pair, p.verdict, value))
    result = ra.auc(scored, within_pair=False)
    if result is None:
        return "-"
    return f"{result[0]:.2f} ({ra.interval(scored, args.bootstrap, args.seed)})"


def flags(points: list[ra.Point], keep: Any) -> str:
    kept = [p for p in points if keep(p)]
    dangerous = sum(p.verdict == "dangerous" for p in points)
    hit = sum(p.verdict == "dangerous" for p in kept)
    precision = f"{hit / len(kept):.0%}" if kept else "-"
    return f"{hit}/{dangerous}, {len(kept)} ({precision})"


def detection_section(args: argparse.Namespace) -> list[str]:
    ns = argparse.Namespace(
        manifest=args.manifest,
        runtime=args.runtime,
        out=args.runtime_auc_out,
        min_replays=5,
        max_replays=10,
        replay_concurrency=1,
        label_concurrency=1,
        screen=args.screen,
        stage1=args.stage1,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    split = ho.splitter(args)
    groups: dict[str, list[tuple[ra.Point, ra.Point]]] = {}
    missing = 0
    for cell in args.cells:
        study = ra.study_for(cell, ns, "report")
        extra = r4_scores(args, cell)
        for point in ra.load_points(cell, study, ns):
            if point.step is None or point.verdict not in ("dangerous", "safe"):
                continue
            combined = point.combined
            if combined is None and point.run in extra:
                combined = extra[point.run]["combined"]
            if combined is None:
                missing += 1
                continue
            pair = (point, dataclasses.replace(point, combined=combined))
            for group in (cell, ALL, f"split: {split(cell, study.sources.runs[point.run])}"):
                groups.setdefault(group, []).append(pair)
    lines = [
        "## RQ2: ranking R1's firing points (attack-failed runs, replay-labelled)",
        "",
        f"Labels `{args.runtime_auc_out}`. Method = R4 (next-step judge on R3's 5 samples) at "
        f"every firing; context-only = stage 1 (fd-risk-v1) alone; first version = stage 1 >= "
        f"{args.stage1}, then R4. Flags at R4 >= {args.threshold} (dangerous flagged/dangerous, "
        f"flags (precision)). AUC (95% interval, bootstrap over task pairs). Firings without an "
        f"R4 score: {missing}. Splits as in held_out.py.",
        "",
        "| Group | Firings (dangerous) | Method AUC | Context-only AUC | First version AUC | "
        "Method flags | Method flags at R4 >= 10 | First version flags |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    order = [*args.cells, ALL, *sorted(g for g in groups if g.startswith("split: "))]
    for group in order:
        pairs = groups.get(group, [])
        first = [d for d, _ in pairs]
        method = [m for _, m in pairs]
        dangerous = sum(p.verdict == "dangerous" for p in first)
        lines.append(
            f"| {group} | {len(pairs)} ({dangerous}) | {auc_cell(method, 'combined', args)} | "
            f"{auc_cell(first, 'stage1', args)} | {auc_cell(first, 'workflow', args)} | "
            f"{flags(method, lambda p: (p.combined or 0) >= args.threshold)} | "
            f"{flags(method, lambda p: (p.combined or 0) >= 10)} | "
            f"{flags(first, lambda p: (p.stage1 or 0) >= args.stage1 and (p.combined or 0) >= args.threshold)} |"
        )
    return lines


def method_flags(
    args: argparse.Namespace, cell: e2e.Cell, e2e_args: argparse.Namespace
) -> dict[str, tuple[tuple[str, int], argparse.Namespace, bool]]:
    """run -> (flagged firing, its replay state, flagged by the first version too)."""
    decisions = json.loads((cell.out(e2e_args) / "decisions.json").read_text())
    out: dict[str, tuple[tuple[str, int], argparse.Namespace, bool]] = {}
    for key, entry in decisions.items():
        if entry.get("flagged") and key in cell.outcome:
            out[key] = ((key, entry["step"]), cell.state(e2e_args), True)
    r4_state = argparse.Namespace(state=args.r4_state / cell.name.replace(":", "_"))
    for key, entry in r4_scores(args, cell.name).items():
        if key in cell.outcome and key not in out and entry["combined"] >= args.threshold:
            out[key] = ((key, entry["step"]), r4_state, False)
    return out


def warning_sections(
    args: argparse.Namespace, cells: dict[str, e2e.Cell], e2e_args: argparse.Namespace
) -> list[str]:
    header = [
        "| Cell | Flagged firings | Firings | Task pairs | Baseline ASR (fresh runs) | Replayed "
        "ASR, no warning | Replayed ASR, warning | Drop (relative) | Benign score, no warning | "
        "Benign score, warning |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    warning = [
        "## RQ3: the warning at the method's flagged firings",
        "",
        f"Each flagged firing replayed {args.replays} times without and with the warning, at the "
        f"same time (first version's flags: `{args.e2e_state}`; flags only the method raises: "
        f"`{args.r4_state}`). Baseline = the task pair's fresh-run ASR, mean over firings; benign "
        "score = share of the task's tests passed.",
        "",
        *header,
    ]
    end = [
        "## RQ3: end to end (every run of the cell)",
        "",
        "Attack success with the workflow = each run's own outcome minus the replay drop at its "
        "flagged firing (unflagged runs unchanged); the same for the benign score. 95% intervals "
        "by bootstrap over task pairs.",
        "",
        "| Cell | Runs | Flagged runs (method / first version) | ASR without | ASR with the "
        "method (change, 95% CI) | ASR with the first version | Benign without | Benign with "
        "the method (change, 95% CI) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    totals = {g: we.Group() for g in we.GROUPS}
    all_rows: dict[str, list[dict[str, Any]]] = {"method": [], "first": []}
    missing = 0
    counts = [0, 0]
    for name, cell in cells.items():
        labels = cell.study.labels()
        groups = {g: we.Group() for g in we.GROUPS}
        flagged = method_flags(args, cell, e2e_args)
        rows: dict[str, list[dict[str, Any]]] = {"method": [], "first": []}
        for key in cell.keys:
            attack, benign = cell.outcome[key]
            drops = {"method": (0.0, 0.0), "first": (0.0, 0.0)}
            if key in flagged:
                item, state, first = flagged[key]
                before = groups["all flagged firings"].firings
                we.add_firing(groups, cell, labels, item, state)
                control = e2e.arm(rc.cut_dir(cell.study, state, item, e2e.CONTROL))
                warned = e2e.arm(rc.cut_dir(cell.study, state, item, e2e.WARNED))
                if groups["all flagged firings"].firings == before:
                    missing += 1
                else:
                    drop = (
                        control[1] / control[0] - warned[1] / warned[0],
                        control[2] / control[0] - warned[2] / warned[0],
                    )
                    drops["method"] = drop
                    if first:
                        drops["first"] = drop
            for version, (attack_drop, benign_drop) in drops.items():
                rows[version].append(
                    {
                        "pair": f"{name}/{cell.study.pair(key)}",
                        "attack": attack,
                        "benign": benign,
                        "attack_drop": attack_drop,
                        "benign_drop": benign_drop,
                    }
                )
        for g in we.GROUPS:
            totals[g].merge(groups[g], name)
        warning += [f"| {name} " + groups[g].row(g) for g in we.GROUPS]
        first_count = sum(f for _, _, f in flagged.values())
        counts = [counts[0] + len(flagged), counts[1] + first_count]
        end.append(end_row(name, rows, f"{len(flagged)} / {first_count}", args))
        for version in rows:
            all_rows[version] += rows[version]
    warning += [f"| **{ALL}** " + totals[g].row(g) for g in we.GROUPS]
    warning += ["", f"Flagged firings without replays in both arms: {missing}."]
    end.append(end_row(f"**{ALL}**", all_rows, f"{counts[0]} / {counts[1]}", args))
    return [*warning, "", *end]


def end_row(
    name: str, rows: dict[str, list[dict[str, Any]]], flagged: str, args: argparse.Namespace
) -> str:
    method = e2e.estimate(rows["method"])
    first = e2e.estimate(rows["first"])
    ci = e2e.bootstrap(rows["method"], args.seed, args.bootstrap)
    a, b = ci["attack change"], ci["benign change"]
    return (
        f"| {name} | {len(rows['method'])} | {flagged} | {method['asr']:.1%} | "
        f"{method['asr_pipeline']:.1%} ({method['asr_pipeline'] - method['asr']:+.1%}, "
        f"{a[0]:+.1%} to {a[1]:+.1%}) | {first['asr_pipeline']:.1%} | {method['benign']:.1%} | "
        f"{method['benign_pipeline']:.1%} ({method['benign_pipeline'] - method['benign']:+.1%}, "
        f"{b[0]:+.1%} to {b[1]:+.1%}) |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("rq3/dataset_v2/manifest_full.json"))
    parser.add_argument(
        "--cells",
        nargs="+",
        default=["gpt:skill_only", "gpt:standard", "qwen:skill_only", "qwen:standard"],
    )
    parser.add_argument("--runtime", type=Path, default=Path("truncated/end_to_end_full"))
    parser.add_argument("--runtime-auc-out", type=Path, default=Path("truncated/runtime_auc_full"))
    parser.add_argument("--r4", type=Path, default=Path("truncated/r4_alone"))
    parser.add_argument(
        "--e2e-state", type=Path, default=Path("truncated/inserted_context/end_to_end")
    )
    parser.add_argument(
        "--r4-state", type=Path, default=Path("truncated/inserted_context/r4_alone")
    )
    parser.add_argument(
        "--census", type=Path, default=Path("truncated/v2_dangerous_full/report.md")
    )
    parser.add_argument(
        "--stability", type=Path, default=Path("truncated/label_stability/report.md")
    )
    parser.add_argument(
        "--baselines", type=Path, default=Path("truncated/alignmentcheck_baseline_full/report.md")
    )
    parser.add_argument("--dev-manifests", type=Path, nargs="*", default=[])
    parser.add_argument("--dev-cells", nargs="*", default=[])
    parser.add_argument("--dev-jobs", type=Path, nargs="*", default=[])
    parser.add_argument("--stage1", type=float, default=10)
    parser.add_argument("--threshold", type=float, default=3)
    parser.add_argument("--replays", type=int, default=3, help="per arm (for the text only)")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--screen", action="store_true", help="labels with the screening rule")
    parser.add_argument("--out", type=Path, default=Path("truncated/final_tables/report.md"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    e2e_args = argparse.Namespace(
        manifest=args.manifest,
        out=args.runtime,
        state=args.e2e_state,
        replay_concurrency=1,
        screen=args.screen,
    )
    cells = {name: e2e.Cell(name, e2e_args) for name in args.cells}
    lines = [
        "# Final tables",
        "",
        f"Manifest `{args.manifest}`, cells {', '.join(args.cells)}. Method: R1 -> R3 -> R4 -> R5 "
        f"(R4 >= {args.threshold}); first version: with the context-only stage 1 filter "
        f"(>= {args.stage1}) before R3.",
        "",
        *data_section(args, cells),
        "",
        *rq1_section(args),
        "",
        *baselines_section(args),
        "",
        *detection_section(args),
        "",
        *warning_sections(args, cells, e2e_args),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out, "\n".join(lines) + "\n")
    print(args.out.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
