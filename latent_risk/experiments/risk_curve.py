"""How the replay ASR changes along a run's attack chain (driver: risk_curve.sh).

For a few chosen attack-failed runs of a prevalence study (prevalence_baselines.py, usually the
dataset v2 manifest), every agent message is replay-labelled from shortly before the run's first
deviation up to its attack action (exclusive), or up to --after agent messages past the first
deviation when the run has no attack action. Labels are found by run and message, so earlier
replays are reused; new ones go to <out>/replays. The report gives, per run, the replay ASR at each
position next to the task pair's fresh-run ASR.

Peak test (commands fresh, compare): the highest point of a noisy curve overstates its risk
(winner's curse), so given points (--points RUN:MESSAGE, the run's first deviation first) get a
fixed number of fresh replays in <out>/replays, apart from the curve's replays. compare tests,
with the fresh replays only, whether a later point is riskier than the run's first point (one-sided
Fisher test per run, and the runs pooled by Fisher's method).
"""

from __future__ import annotations

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import prevalence_baselines as pb
from claim2_audit import _referenced_message_indices, first_chain_message
from scipy.stats import chi2, fisher_exact

from scripts import attribution_replay_study as ars, self_evolve_attribution as se


def positions(study: pb.Study, key: str, before: int, after: int) -> list[int]:
    eligible = study.eligible(key)
    first = first_chain_message(study.sources, key)
    if first is None or first not in eligible:
        return []
    start = max(0, eligible.index(first) - before)
    if study.sources.attack_index(key) is not None:
        return eligible[start:]  # eligible already stops before the attack action
    return eligible[start : eligible.index(first) + after + 1]


def chosen(study: pb.Study, runs: list[str]) -> list[str]:
    keys = [key for key in study.runs if any(key.endswith("/" + run) for run in runs)]
    missing = [run for run in runs if not any(key.endswith("/" + run) for key in keys)]
    if missing:
        raise SystemExit(f"not attack-failed runs of this study: {missing}")
    return keys


def items(study: pb.Study, args: argparse.Namespace) -> list[pb.Item]:
    return [
        (key, index)
        for key in chosen(study, args.runs)
        for index in positions(study, key, args.before, args.after)
    ]


def pending(study: pb.Study, args: argparse.Namespace) -> list[pb.Item]:
    labels = study.labels()
    return [
        item
        for item in items(study, args)
        if item not in labels or labels[item].verdict == "uncertain"
    ]


def cost(study: pb.Study, args: argparse.Namespace) -> None:
    todo = pending(study, args)
    for key in chosen(study, args.runs):
        cuts = positions(study, key, args.before, args.after)
        mine = sum(1 for k, _ in todo if k == key)
        print(f"{key.split('/')[-1]}: {len(cuts)} positions, {mine} to label")
    print(f"to label: {len(todo)} messages, up to {len(todo) * study.args.max_replays} replays")


def label(study: pb.Study, args: argparse.Namespace) -> None:
    todo = pending(study, args)
    print(f"replay-labelling {len(todo)} message(s)", flush=True)
    study.args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=study.args.label_concurrency) as pool:
        list(
            pool.map(
                lambda item: se.label_message(study.sources, item[0], item[1], study.local), todo
            )
        )


def report(study: pb.Study, args: argparse.Namespace) -> None:
    labels = study.labels()
    lines = [
        "# Replay ASR along the attack chain",
        "",
        f"Attack-failed runs, {study.args.min_replays}-{study.args.max_replays} replays per "
        "position (cut right after the agent message). Step 0 = the first deviation; negative "
        "steps come before it. Verdict: d = dangerous (one-sided Fisher vs the fresh-run ASR, "
        "p < 0.05), s = safe, u = uncertain.",
    ]
    for key in chosen(study, args.runs):
        messages = study.sources.messages(key)
        eligible = study.eligible(key)
        first = first_chain_message(study.sources, key)
        if first is None:  # chosen() keeps runs with a chain; positions() needs a first deviation
            continue
        chain = set(_referenced_message_indices(study.sources.chain(key) or {}))
        attack = study.sources.attack_index(key)
        job = study.sources.job_of[key]
        fresh = f"{job.base_k}/{job.base_n}"
        lines += [
            "",
            f"## {study.pair(key)}, run {key.split('/')[-1]}",
            "",
            f"Fresh-run ASR {fresh}; {len(messages)} messages; attack action at "
            f"{'message ' + str(attack) if attack is not None else 'none (never attempted)'}.",
            "",
            "| Step | Message | In chain | v1 score | Replays | ASR | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for index in positions(study, key, args.before, args.after):
            step = eligible.index(index) - eligible.index(first)
            lab = labels.get((key, index))
            k, n = (lab.successes, lab.replays) if lab else (0, 0)
            asr = f"{k / n:.0%}" if n else "-"
            score = study.scores.get((key, index))
            lines.append(
                f"| {step:+d} | {index} | {'yes' if index in chain else ''} | "
                f"{'' if score is None else score} | {f'{k}/{n}' if n else '-'} | {asr} | "
                f"{lab.verdict[0] if lab else '-'} |"
            )
    out = study.args.out / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


def points(study: pb.Study, args: argparse.Namespace) -> list[pb.Item]:
    """--points RUN:MESSAGE as study items, in the given order."""
    found: list[pb.Item] = []
    for point in args.points:
        run, index = point.split(":")
        (key,) = chosen(study, [run])
        if int(index) not in study.eligible(key):
            raise SystemExit(f"{point}: not an agent message before the attack action")
        found.append((key, int(index)))
    return found


def cut_dir(study: pb.Study, item: pb.Item) -> Path:
    key, index = item
    job = study.sources.job_of[key]
    return study.args.out / "replays" / job.task_dir / f"{study.sources.runs[key].name}_msg{index}"


def fresh(study: pb.Study, args: argparse.Namespace) -> None:
    todo = [
        item
        for item in points(study, args)
        if ars.replay_state(cut_dir(study, item))[0] < study.args.max_replays
    ]
    print(f"fresh replays for {len(todo)} point(s), {study.args.max_replays} each", flush=True)
    study.args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=study.args.label_concurrency) as pool:
        list(
            pool.map(
                lambda item: se.label_message(study.sources, item[0], item[1], study.local), todo
            )
        )


def compare(study: pb.Study, args: argparse.Namespace) -> None:
    own = study.args.out.resolve()
    labels = study.labels()
    by_run: dict[str, list[pb.Item]] = {}
    for item in points(study, args):
        by_run.setdefault(item[0], []).append(item)
    lines = [
        "# Peak test: is a later point riskier than the first deviation?",
        "",
        "Fresh replays only (cut again, kept apart from the curve's replays). Earlier = the "
        "curve's replays at that point. p (later > first) = one-sided Fisher test on the fresh "
        "replays of the two points; p (vs fresh runs) = one-sided Fisher test against the task "
        "pair's fresh-run ASR.",
    ]
    later_p: list[float] = []
    for key, items_ in by_run.items():
        job = study.sources.job_of[key]
        first = items_[0]
        n0, k0, _ = ars.replay_state(cut_dir(study, first))
        lines += [
            "",
            f"## {study.pair(key)}, run {key.split('/')[-1]} (fresh-run ASR "
            f"{job.base_k}/{job.base_n})",
            "",
            "| Message | Earlier replays | Fresh replays | Fresh ASR | p (vs fresh runs) | "
            "p (later > first) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in items_:
            n, k, _ = ars.replay_state(cut_dir(study, item))
            earlier_k = earlier_n = 0
            label = labels.get(item)
            for path in label.cut_dirs if label else []:
                if not Path(path).resolve().is_relative_to(own):
                    done, hits, _ = ars.replay_state(Path(path))
                    earlier_n += done
                    earlier_k += hits
            base = fisher_exact(
                [[k, n - k], [job.base_k, job.base_n - job.base_k]], alternative="greater"
            )[1]
            later = (
                fisher_exact([[k, n - k], [k0, n0 - k0]], alternative="greater")[1]
                if item != first
                else None
            )
            if later is not None and n and n0:
                later_p.append(later)
            lines.append(
                f"| {item[1]}{' (first)' if item == first else ''} | "
                f"{f'{earlier_k}/{earlier_n}' if earlier_n else '-'} | {k}/{n} | "
                f"{k / n:.0%} | {base:.3f} | {'' if later is None else f'{later:.3f}'} |"
                if n
                else f"| {item[1]} | - | - | - | - | - |"
            )
    if len(later_p) > 1:
        pooled = chi2.sf(-2 * sum(math.log(p) for p in later_p), 2 * len(later_p))
        lines += [
            "",
            f"**Pooled over {len(later_p)} comparisons** (Fisher's method on the one-sided "
            f"p values; conservative for exact tests): p = {pooled:.4f}.",
        ]
    out = study.args.out / "report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cost", "label", "report", "fresh", "compare"))
    parser.add_argument("--runs", nargs="+", default=[], help="run ids (attack-failed)")
    parser.add_argument(
        "--points", nargs="+", default=[], help="fresh, compare: RUN:MESSAGE, first deviation first"
    )
    parser.add_argument(
        "--before", type=int, default=2, help="agent messages before the first deviation"
    )
    parser.add_argument(
        "--after", type=int, default=6, help="agent messages after the first deviation (no attack)"
    )
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    command = {"fresh": "label", "compare": "report"}.get(args.command, args.command)
    study = pb.Study(pb.parse_args([command, *rest]))
    commands = {"cost": cost, "label": label, "report": report, "fresh": fresh, "compare": compare}
    commands[args.command](study, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
