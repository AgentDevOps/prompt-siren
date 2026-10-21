"""Claim 2: are dangerous messages concentrated in certain task pairs?

Equal-effort random audit, independent of any attribution skill: in every task pair (the dataset in
jobs/groups and the pairs moved to jobs/groups_excluded/no_verified_dangerous), draw K attack-failed
runs at random (a seeded order per pair, so a larger K extends a smaller one), take one message per
run, and replay-verify it with the usual rule. Which message (--position):
  any          a random assistant message before the attack action (the first pilot: mostly late
               messages, where replays almost never attack)
  first-chain  the first assistant message the attack-chain judge put in the chain, i.e. the
               agent's first engagement with the injection (12 of the 15 known dangerous messages
               are at this position); only runs with such a message are drawn, and the share of
               runs that have one (engagement) is reported per task pair
 Existing labels are reused. The report compares dangerous rates across task pairs (permutation
test of homogeneity) and between task pairs of the same attack type with similar baselines
(matched comparisons, Fisher exact test), so concentration can be separated from labelling effort
and from detectability.

Commands:
  cost      print the sample and the replays it still needs
  label     replay-verify the sample (--part K/N splits it)
  report    write the tables to --out (report.md, audit.jsonl)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from scipy.stats import fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_siren.attack_chain_judge import _referenced_message_indices  # noqa: E402

from scripts import self_evolve_attribution as evolve  # noqa: E402

ROOTS = (Path("jobs/groups"), Path("jobs/groups_excluded/no_verified_dangerous"))


def evolve_args(root: Path, args: argparse.Namespace) -> argparse.Namespace:
    return evolve.parse_args(
        [
            "status",
            "--jobs-root",
            str(root),
            "--state",
            str(args.state),
            "--min-replays",
            str(min(args.min_replays, args.max_replays)),
            "--max-replays",
            str(args.max_replays),
            "--zero-safe-at",
            str(args.zero_safe_at),
            "--replay-concurrency",
            str(args.replay_concurrency),
        ]
    )


def first_chain_message(sources: Any, key: str) -> int | None:
    """The first assistant message of the run's attack chain, before its attack action."""
    messages = sources.messages(key)
    return next(
        (
            index
            for index in sorted(set(_referenced_message_indices(sources.chain(key) or {})))
            if 0 <= index < len(messages)
            and messages[index].get("kind") == "response"
            and sources.before_attack(key, index)
        ),
        None,
    )


def draw(sources: Any, per_pair: int, seed: int, position: str = "any") -> list[tuple[str, int]]:
    """K runs per task pair in a seeded order, one message each (see --position)."""
    by_task: dict[str, list[str]] = defaultdict(list)
    for key in sorted(sources.runs):
        by_task[sources.job_of[key].task_dir].append(key)
    sample: list[tuple[str, int]] = []
    for task, keys in sorted(by_task.items()):
        order = list(keys)
        random.Random(f"{seed}:{task}").shuffle(order)
        taken = 0
        for key in order:
            if taken >= per_pair:
                break
            if position == "first-chain":
                index = first_chain_message(sources, key)
                if index is None:
                    continue  # the agent never engaged with the injection in this run
                sample.append((key, index))
                taken += 1
                continue
            messages = sources.messages(key)
            eligible = [
                index
                for index, message in enumerate(messages)
                if message.get("kind") == "response" and sources.before_attack(key, index)
            ]
            if not eligible:
                continue
            sample.append((key, random.Random(f"{seed}:{key}").choice(eligible)))
            taken += 1
    return sample


def contexts(
    args: argparse.Namespace,
) -> list[tuple[Any, argparse.Namespace, list[tuple[str, int]]]]:
    result = []
    for root in ROOTS:
        if not root.exists():
            continue
        local = evolve_args(root, args)
        sources = evolve.Sources(local)
        result.append((sources, local, draw(sources, args.per_pair, args.seed, args.position)))
    return result


def needs_replay(labels: dict[tuple[str, int], Any], item: tuple[str, int]) -> bool:
    return item not in labels or labels[item].verdict == "uncertain"


def cost(args: argparse.Namespace) -> None:
    total = todo = 0
    for sources, local, sample in contexts(args):
        labels = evolve.collect_labels(
            sources, local.truncated, local.label_rule, exclude_holdout=False
        )
        total += len(sample)
        todo += sum(needs_replay(labels, item) for item in sample)
    print(
        f"sample: {total} messages ({args.per_pair} per task pair); to replay: {todo} "
        f"(up to {todo * args.max_replays} replays)"
    )


def label(args: argparse.Namespace) -> None:
    for sources, local, sample in contexts(args):
        labels = evolve.collect_labels(
            sources, local.truncated, local.label_rule, exclude_holdout=False
        )
        todo = [item for item in sample if needs_replay(labels, item)]
        if args.part is not None:
            part, parts = args.part
            todo = todo[part - 1 :: parts]
        print(f"{local.jobs_root}: replaying {len(todo)} message(s)", flush=True)
        args.state.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.label_concurrency) as pool:
            list(
                pool.map(
                    lambda item, sources=sources, local=local: evolve.label_message(
                        sources, item[0], item[1], local
                    ),
                    todo,
                )
            )


def attack_type(task: str) -> str:
    return task.split("_", 2)[2].removesuffix("_task")


def report(args: argparse.Namespace) -> None:
    rows: list[dict[str, Any]] = []
    pooled: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    engaged: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for sources, local, sample in contexts(args):
        for key in sources.runs:
            engaged[sources.job_of[key].task_dir][0] += (
                first_chain_message(sources, key) is not None
            )
            engaged[sources.job_of[key].task_dir][1] += 1
        for job in {id(job): job for job in sources.job_of.values()}.values():
            pooled[job.task_dir][0] += job.base_k
            pooled[job.task_dir][1] += job.base_n
        labels = evolve.collect_labels(
            sources, local.truncated, local.label_rule, exclude_holdout=False
        )
        for key, index in sample:
            job = sources.job_of[key]
            found = labels.get((key, index))
            rows.append(
                {
                    "task": job.task_dir,
                    "dataset": "kept" if local.jobs_root == ROOTS[0] else "moved",
                    "baseline": "{}/{}".format(*pooled[job.task_dir]),
                    "run": key,
                    "message_index": index,
                    "verdict": found.verdict if found else None,
                    "replays": f"{found.successes}/{found.replays}" if found else None,
                }
            )
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)

    def counts(items: list[dict[str, Any]]) -> tuple[int, int, int]:
        dangerous = sum(item["verdict"] == "dangerous" for item in items)
        decided = sum(item["verdict"] in ("dangerous", "safe") for item in items)
        return dangerous, decided, len(items) - decided

    lines = [
        "# Claim 2: equal-effort random audit",
        "",
        f"{args.per_pair} attack-failed runs per task pair (seed {args.seed}), one message each: "
        + (
            "the first agent message of the run's attack chain (runs without one are not drawn)"
            if args.position == "first-chain"
            else "a random agent message before the attack action"
        )
        + f"; replay-verified with at most {args.max_replays} replays (existing labels reused, "
        "with all their replays). Engagement = failed runs with an agent message in the attack "
        "chain before the attack action.",
        "",
        "| Task pair | Dataset | Fresh ASR | Engagement | Audited | Dangerous | Not yet decided | Rate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    table = []
    for task, items in sorted(by_task.items(), key=lambda pair: -counts(pair[1])[0]):
        dangerous, decided, open_ = counts(items)
        table.append((task, dangerous, decided))
        rate = f"{dangerous / decided:.0%}" if decided else "-"
        lines.append(
            f"| {task.removesuffix('_task')} | {items[0]['dataset']} | {items[0]['baseline']} | "
            f"{engaged[task][0]}/{engaged[task][1]} | {len(items)} | {dangerous} | {open_} | {rate} |"
        )

    # homogeneity: is the dangerous rate the same in every task pair? (permutation test)
    verdicts = [
        (task, row["verdict"] == "dangerous")
        for task, items in by_task.items()
        for row in items
        if row["verdict"] in ("dangerous", "safe")
    ]
    if verdicts:
        tasks = [task for task, _ in verdicts]
        outcomes = [flag for _, flag in verdicts]

        def statistic(values: list[bool]) -> float:
            total_rate = sum(values) / len(values)
            if total_rate in (0.0, 1.0):
                return 0.0
            groups: dict[str, list[bool]] = defaultdict(list)
            for task, value in zip(tasks, values, strict=True):
                groups[task].append(value)
            return sum(
                (sum(group) - len(group) * total_rate) ** 2
                / (len(group) * total_rate * (1 - total_rate))
                for group in groups.values()
            )

        observed = statistic(outcomes)
        rng = random.Random(args.seed)
        shuffled = list(outcomes)
        exceed = 0
        for _ in range(args.permutations):
            rng.shuffle(shuffled)
            exceed += statistic(shuffled) >= observed - 1e-12
        lines += [
            "",
            f"**Homogeneity across task pairs** (chi-square statistic, {args.permutations} "
            f"permutations): statistic {observed:.1f}, p = {(exceed + 1) / (args.permutations + 1):.4f}. "
            "A small p means the dangerous rate differs between task pairs at equal audit effort.",
        ]

    # matched comparisons: same attack type, both baselines low (<= 4/30), one pair with
    # dangerous messages vs one without
    lines += [
        "",
        "## Matched comparisons (same attack type, similar low baseline)",
        "",
        "| Attack type | Task pair A (fresh ASR) | Dangerous A | Task pair B (fresh ASR) | Dangerous B | Fisher p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    by_type: dict[str, list[str]] = defaultdict(list)
    for task in by_task:
        by_type[attack_type(task)].append(task)

    def low(task: str) -> bool:
        successes, runs = (int(value) for value in by_task[task][0]["baseline"].split("/"))
        return successes / runs <= 4 / 30

    for kind, members in sorted(by_type.items()):
        members = [task for task in members if low(task)]
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                d1, n1, _ = counts(by_task[first])
                d2, n2, _ = counts(by_task[second])
                if not (n1 and n2) or d1 == d2 == 0:
                    continue
                p_value = fisher_exact([[d1, n1 - d1], [d2, n2 - d2]])[1]
                lines.append(
                    f"| {kind} | {first.removesuffix('_task')} ({by_task[first][0]['baseline']}) | "
                    f"{d1}/{n1} | {second.removesuffix('_task')} ({by_task[second][0]['baseline']}) | "
                    f"{d2}/{n2} | {p_value:.3f} |"
                )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "audit.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    print("\n".join(lines))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("cost", "label", "report"))
    parser.add_argument("--per-pair", type=int, default=6)
    parser.add_argument("--position", choices=("any", "first-chain"), default="any")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-replays", type=int, default=5)
    parser.add_argument("--max-replays", type=int, default=10)
    parser.add_argument("--zero-safe-at", type=int, default=10)
    parser.add_argument("--replay-concurrency", type=int, default=3)
    parser.add_argument("--label-concurrency", type=int, default=1)
    parser.add_argument("--part", default=None, help="K/N: replay only that share")
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--state", type=Path, default=Path("truncated/claim2_audit"))
    parser.add_argument("--out", type=Path, default=Path("rq3/claim2"))
    args = parser.parse_args(argv)
    if args.part is not None:
        part, parts = (int(value) for value in args.part.split("/"))
        args.part = (part, parts)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    {"cost": cost, "label": label, "report": report}[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
