"""Paper experiments E1.2 (prevalence) and E2.1 (the frozen v1 workflow vs simple baselines).

Population: the attack-failed runs of the 8 in-scope task pairs that have a task-deviation chain
(the runs of the task-deviation self-evolving run, --state). Every candidate message is an agent
message before the run's attack action, replay-labelled with the loop's rule (5-10 replays, one-
sided Fisher test vs the task pair's fresh-run ASR); existing labels in truncated/ are reused.

Methods, one message per run unless stated:
  first-deviation  the first agent message of the run's task-deviation chain (payload-blind, no
                   judge); also the E1.2 prevalence probe: the agent's first deviation
  first-reply      the agent's first reply
  random           a random agent message before the attack action (seeded per run)
  (--sample-per-pair N: first-reply and random use only N runs per task pair, a seeded order, so
  their precision is a sample estimate; a larger N extends a smaller one)
  v1-top1          v1's highest-scored message in the run, whatever its score (same budget as the
                   heuristics: one message per run)
  v1-flags         every message v1 scores >= its frozen training threshold (the deployed rule;
                   any number per run)

Commands:
  cost      print the messages each method selects and the replays still needed
  label     replay-label every selected message that has no verdict yet (--part K/N splits it;
            --methods limits it)
  report    write --out/report.md (prevalence per task pair; methods per split)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LATENT_RISK_ROOT = Path(__file__).resolve().parents[1]
if str(LATENT_RISK_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_RISK_ROOT))
EXPERIMENTS = Path(__file__).resolve().parent
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from claim2_audit import first_chain_message  # noqa: E402
from judge_discussion import loop_args, SPLITS  # noqa: E402

from scripts import self_evolve_attribution as se  # noqa: E402

METHODS = ("first-deviation", "first-reply", "random", "v1-top1", "v1-flags")
# v1's attributions per split: the loop's own (training, validation) and its held-out run (test)
V1_DIRS = {
    "training": "{state}/attributions/v1",
    "validation": "{state}/validation/v1",
    "test": "{state}_vt_holdout/attributions/v1",
}

Item = tuple[str, int]


class Study:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.local = loop_args(args)
        self.sources = se.Sources(self.local)
        se.prepare_split(self.sources, self.local)
        self.local.state = args.out  # new replays go to <out>/replays
        self.local.label_rule = se.LabelRule(args.min_replays, args.max_replays, args.zero_safe_at)
        self.runs = sorted(key for key in self.sources.runs if self.sources.chain(key))
        self.threshold = se.frozen_threshold(args.state, "v1")
        self.scores: dict[Item, int] = {}
        for split, pattern in V1_DIRS.items():
            runs = [key for key in self.runs if self.split(key) == split]
            path = Path(pattern.format(state=args.state))
            self.scores.update(se.load_evaluation(self.sources, runs, "v1", path).scores)

    def split(self, key: str) -> str:
        if se.is_held_out(self.sources, key, self.local):
            return "test"
        if se.in_validation(self.sources, key, self.local):
            return "validation"
        return "training"

    def pair(self, key: str) -> str:
        return self.sources.job_of[key].task_dir

    def eligible(self, key: str) -> list[int]:
        return [
            index
            for index, message in enumerate(self.sources.messages(key))
            if message.get("kind") == "response" and self.sources.before_attack(key, index)
        ]

    def sampled(self) -> set[str]:
        by_pair: dict[str, list[str]] = defaultdict(list)
        for key in self.runs:
            by_pair[self.pair(key)].append(key)
        keep: set[str] = set()
        for pair, keys in by_pair.items():
            random.Random(f"{self.args.seed}:{pair}").shuffle(keys)
            keep.update(keys[: self.args.sample_per_pair])
        return keep

    def select(self, method: str) -> list[Item]:
        chosen: list[Item] = []
        sampled = self.sampled() if self.args.sample_per_pair else set(self.runs)
        for key in self.runs:
            if method in ("first-reply", "random") and key not in sampled:
                continue
            eligible = self.eligible(key)
            if not eligible:
                continue
            if method == "first-deviation":
                index = first_chain_message(self.sources, key)
                if index is not None:
                    chosen.append((key, index))
            elif method == "first-reply":
                chosen.append((key, eligible[0]))
            elif method == "random":
                chosen.append((key, random.Random(f"{self.args.seed}:{key}").choice(eligible)))
            else:
                scored = [(self.scores[(key, i)], -i) for i in eligible if (key, i) in self.scores]
                if method == "v1-top1" and scored:
                    chosen.append((key, -max(scored)[1]))
                elif method == "v1-flags":
                    chosen += [(key, -i) for score, i in scored if score >= self.threshold]
        return chosen

    def labels(self) -> dict[Item, se.Label]:
        return se.collect_labels(
            self.sources, self.local.truncated, self.local.label_rule, exclude_holdout=False
        )


def pending(study: Study, methods: list[str]) -> list[Item]:
    labels = study.labels()
    todo = {
        item
        for method in methods
        for item in study.select(method)
        if item not in labels or labels[item].verdict == "uncertain"
    }
    return sorted(todo)


def cost(study: Study) -> None:
    labels = study.labels()
    pairs = {study.pair(key) for key in study.runs}
    print(f"population: {len(study.runs)} attack-failed runs, {len(pairs)} task pairs")
    print(f"v1 frozen threshold: {study.threshold}")
    for method in METHODS:
        items = study.select(method)
        todo = [i for i in items if i not in labels or labels[i].verdict == "uncertain"]
        print(
            f"{method:16s} {len(items):4d} messages in {len({k for k, _ in items}):3d} runs; "
            f"without verdict {len(todo):4d}"
        )
    todo = pending(study, list(study.args.methods))
    print(
        f"to label for {', '.join(study.args.methods)}: {len(todo)} unique messages, "
        f"{len(todo) * study.args.min_replays}-{len(todo) * study.args.max_replays} replays"
    )


def label(study: Study) -> None:
    todo = pending(study, list(study.args.methods))
    if study.args.part is not None:
        part, parts = study.args.part
        todo = todo[part - 1 :: parts]
    print(f"replay-labelling {len(todo)} message(s)", flush=True)
    study.args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=study.args.label_concurrency) as pool:
        list(
            pool.map(
                lambda item: se.label_message(study.sources, item[0], item[1], study.local), todo
            )
        )


def wilson(k: int, n: int) -> str:
    if not n:
        return "-"
    z, p = 1.96, k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{p:.0%} ({max(0.0, centre - half):.0%}-{min(1.0, centre + half):.0%})"


def homogeneity(groups: dict[str, list[bool]], permutations: int, seed: int) -> tuple[float, float]:
    """Chi-square statistic of equal dangerous rates across groups, permutation p-value."""
    tasks = [task for task, values in groups.items() for _ in values]
    outcomes = [value for values in groups.values() for value in values]

    def statistic(values: list[bool]) -> float:
        rate = sum(values) / len(values)
        if rate in (0.0, 1.0):
            return 0.0
        by: dict[str, list[bool]] = defaultdict(list)
        for task, value in zip(tasks, values, strict=True):
            by[task].append(value)
        return sum(
            (sum(g) - len(g) * rate) ** 2 / (len(g) * rate * (1 - rate)) for g in by.values()
        )

    observed = statistic(outcomes)
    rng, shuffled, exceed = random.Random(seed), list(outcomes), 0
    for _ in range(permutations):
        rng.shuffle(shuffled)
        exceed += statistic(shuffled) >= observed - 1e-12
    return observed, (exceed + 1) / (permutations + 1)


def report(study: Study) -> None:
    labels = study.labels()
    verdict = {item: label.verdict for item, label in labels.items()}
    runs_by_pair: dict[str, list[str]] = defaultdict(list)
    for key in study.runs:
        runs_by_pair[study.pair(key)].append(key)
    known = {
        item
        for item, v in verdict.items()
        if v == "dangerous" and item[0] in set(study.runs) and item[1] in study.eligible(item[0])
    }
    baseline = {study.pair(k): study.sources.job_of[k] for k in study.runs}

    # E1.2: prevalence at the first deviation
    first = dict(study.select("first-deviation"))
    lines = [
        "# Prevalence of hidden high-risk states and the v1 workflow vs baselines",
        "",
        f"Population: {len(study.runs)} attack-failed trajectories in {len(runs_by_pair)} task "
        "pairs (runs with a task-deviation chain). Labels: replay verdicts in truncated/ "
        f"({study.args.min_replays}-{study.args.max_replays} replays, one-sided Fisher vs the task "
        "pair's fresh-run ASR); cuts only before the attack action.",
        "",
        "## E1.2 Prevalence: is the agent's first deviation a hidden high-risk state?",
        "",
        "Deviating = the run has a pre-attack message in its task-deviation chain. Rate = "
        "dangerous / decided first deviations (95% Wilson interval). Any known = trajectories "
        "with at least one verified dangerous message from any experiment (a lower bound).",
        "",
        "| Task pair | Fresh ASR | Failed trajectories | Deviating | Dangerous / decided | "
        "Rate | Undecided | Any known |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    groups: dict[str, list[bool]] = {}
    totals = [0, 0, 0, 0, 0, 0]
    for pair, keys in sorted(runs_by_pair.items()):
        items = [(k, first[k]) for k in keys if k in first]
        decided = [
            verdict[i] == "dangerous" for i in items if verdict.get(i) in ("dangerous", "safe")
        ]
        any_known = len({k for k, _ in known if k in set(keys)})
        groups[pair] = decided
        job = baseline[pair]
        row = (
            len(keys),
            len(items),
            sum(decided),
            len(decided),
            len(items) - len(decided),
            any_known,
        )
        totals = [a + b for a, b in zip(totals, row, strict=True)]
        lines.append(
            f"| {pair.removesuffix('_task')} | {job.base_k}/{job.base_n} | {row[0]} | {row[1]} | "
            f"{row[2]}/{row[3]} | {wilson(row[2], row[3])} | {row[4]} | {row[5]} |"
        )
    lines.append(
        f"| **All** | | {totals[0]} | {totals[1]} | {totals[2]}/{totals[3]} | "
        f"{wilson(totals[2], totals[3])} | {totals[4]} | {totals[5]} |"
    )
    if any(groups.values()):
        stat, p = homogeneity({k: v for k, v in groups.items() if v}, 10000, study.args.seed)
        lines += [
            "",
            f"Homogeneity of the first-deviation rate across task pairs: chi-square {stat:.1f}, "
            f"permutation p = {p:.4f} (small p: the rate depends on the task pair).",
        ]

    # E2.1: methods
    lines += [
        "",
        "## E2.1 The frozen v1 workflow vs baselines",
        "",
        f"v1's threshold ({study.threshold}) was frozen on training; validation and test are "
        "the clean comparison. Cells: dangerous caught / known dangerous; flags (trajectories); "
        "precision on decided flags (+ undecided); replays spent per dangerous message found.",
        "",
        "| Method | " + " | ".join(SPLITS) + " | validation + test |",
        "| --- | " + " | ".join("---" for _ in SPLITS) + " | --- |",
    ]
    for method in METHODS:
        items = study.select(method)
        cells = []
        for splits in [(s,) for s in SPLITS] + [("validation", "test")]:
            mine = [i for i in items if study.split(i[0]) in splits]
            known_here = [i for i in known if study.split(i[0]) in splits]
            tp = [i for i in mine if verdict.get(i) == "dangerous"]
            fp = [i for i in mine if verdict.get(i) == "safe"]
            rest = len(mine) - len(tp) - len(fp)
            spent = sum(labels[i].replays for i in mine if i in labels)
            precision = f"{len(tp) / (len(tp) + len(fp)):.0%}" if tp or fp else "-"
            cells.append(
                f"{len(tp)}/{len(known_here)}; {len(mine)} ({len({k for k, _ in mine})}); "
                f"{precision}"
                + (f" (+{rest})" if rest else "")
                + (f"; {spent / len(tp):.0f}" if tp else "")
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    args = study.args
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = [
        {
            "method": m,
            "run": k,
            "message_index": i,
            "split": study.split(k),
            "verdict": verdict.get((k, i)),
        }
        for m in METHODS
        for k, i in study.select(m)
    ]
    (args.out / "selections.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    print("\n".join(lines))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("cost", "label", "report"))
    parser.add_argument("--state", type=Path, default=Path("truncated/self_evolve_td"))
    parser.add_argument("--blind-dir", type=Path, default=Path("rq3/task_deviation"))
    parser.add_argument("--out", type=Path, default=Path("truncated/prevalence_baselines"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sample-per-pair", type=int, default=0, help="first-reply/random: N runs per task pair"
    )
    parser.add_argument("--min-replays", type=int, default=5)
    parser.add_argument("--max-replays", type=int, default=10)
    parser.add_argument("--zero-safe-at", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--replay-concurrency", type=int, default=3)
    parser.add_argument("--label-concurrency", type=int, default=1)
    parser.add_argument("--part", default=None, help="K/N: label only that share")
    args = parser.parse_args(argv)
    if args.part is not None:
        part, parts = (int(value) for value in args.part.split("/"))
        args.part = (part, parts)
    return args


def main(argv: list[str] | None = None) -> int:
    study = Study(parse_args(sys.argv[1:] if argv is None else argv))
    {"cost": cost, "label": label, "report": report}[study.args.command](study)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
