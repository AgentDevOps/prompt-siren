#!/usr/bin/env python3
"""Remove existing attack-chain attribution output so it can be regenerated.

This is a destructive operation. By default it only lists what would be removed;
pass --apply to actually delete the files. Never touches attack_chain_judge.json,
attack_chain_labeled.json, execution.json, or any other chain/coding output.

Examples:
    # Preview everything found under a job group (no files are touched).
    uv run python scripts/clean_attack_chain_attribution.py jobs/my_job

    # Delete only trajectories that failed to attribute.
    uv run python scripts/clean_attack_chain_attribution.py jobs/my_job \\
        --status failed --apply

    # Delete stale attribution left over from a run that predates --failed-only.
    uv run python scripts/clean_attack_chain_attribution.py jobs/my_job \\
        --only-succeeded-attacks --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

AttributionStatusFilter = Literal["ok", "no_candidate", "insufficient_context", "failed"]
AttackOutcomeFilter = Literal["succeeded", "failed", "any"]

ATTRIBUTION_JSON_NAME = "attack_chain_attribution.json"
ATTRIBUTION_MARKDOWN_NAME = "attack_chain_attribution.md"
ATTRIBUTION_SUMMARY_NAME = "attack_chain_attribution_summary.md"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def attack_already_succeeded_per_outcome(execution_path: Path) -> bool | None:
    """Read the attack outcome from result.json; never inferred from attribution output."""
    result_path = execution_path.with_name("result.json")
    if not result_path.exists():
        return None
    try:
        attack_score = _load_json(result_path).get("attack_score")
    except (OSError, ValueError):
        return None
    return None if attack_score is None else bool(attack_score)


def find_attribution_json_paths(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for input_path in inputs:
        if input_path.is_file() and input_path.name == ATTRIBUTION_JSON_NAME:
            paths.add(input_path.resolve())
        elif input_path.is_dir():
            paths.update(path.resolve() for path in input_path.rglob(ATTRIBUTION_JSON_NAME))
        else:
            raise FileNotFoundError(f"No attribution file or directory found at {input_path}")
    return sorted(paths)


def find_summary_paths(inputs: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for input_path in inputs:
        if input_path.is_file() and input_path.name == ATTRIBUTION_SUMMARY_NAME:
            paths.add(input_path.resolve())
        elif input_path.is_dir():
            paths.update(path.resolve() for path in input_path.rglob(ATTRIBUTION_SUMMARY_NAME))
    return sorted(paths)


def matches_filters(
    json_path: Path,
    *,
    statuses: set[str] | None,
    outcome_filter: AttackOutcomeFilter,
) -> bool:
    """Read-only classification; never mutates or removes anything itself."""
    if statuses is not None:
        try:
            data = _load_json(json_path)
        except (OSError, ValueError) as exc:
            print(f"warning: could not read {json_path}, skipping: {exc}", file=sys.stderr)
            return False
        if data.get("status") not in statuses:
            return False

    if outcome_filter != "any":
        execution_path = json_path.with_name("execution.json")
        succeeded = attack_already_succeeded_per_outcome(execution_path)
        if outcome_filter == "succeeded" and succeeded is not True:
            return False
        if outcome_filter == "failed" and succeeded is not False:
            return False

    return True


def plan_removals(
    inputs: list[Path],
    *,
    statuses: set[str] | None,
    outcome_filter: AttackOutcomeFilter,
    include_summaries: bool,
) -> list[Path]:
    """Return every path that would be removed; callers decide whether to apply it."""
    removals: list[Path] = []
    for json_path in find_attribution_json_paths(inputs):
        if not matches_filters(json_path, statuses=statuses, outcome_filter=outcome_filter):
            continue
        removals.append(json_path)
        md_path = json_path.with_name(ATTRIBUTION_MARKDOWN_NAME)
        if md_path.exists():
            removals.append(md_path)
    if include_summaries:
        removals.extend(find_summary_paths(inputs))
    return sorted(dict.fromkeys(removals))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="One or more attack_chain_attribution.json files or directories to scan recursively.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        choices=("ok", "no_candidate", "insufficient_context", "failed"),
        help="Only remove attribution output with this status. Repeatable. Default: any status.",
    )
    outcome_group = parser.add_mutually_exclusive_group()
    outcome_group.add_argument(
        "--only-succeeded-attacks",
        action="store_const",
        const="succeeded",
        dest="outcome_filter",
        help=(
            "Only remove attribution output for trajectories whose result.json confirms the "
            "attack succeeded (e.g. stale output from a run that predated --failed-only)."
        ),
    )
    outcome_group.add_argument(
        "--only-failed-attacks",
        action="store_const",
        const="failed",
        dest="outcome_filter",
        help="Only remove attribution output for trajectories whose result.json confirms the attack failed.",
    )
    parser.set_defaults(outcome_filter="any")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not also remove attack_chain_attribution_summary.md files found under the inputs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete the files. Without this flag, only lists what would be removed.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    statuses = set(args.status) if args.status else None
    removals = plan_removals(
        args.inputs,
        statuses=statuses,
        outcome_filter=args.outcome_filter,
        include_summaries=not args.no_summary,
    )

    if not removals:
        print("No matching attribution output found.")
        return 0

    for path in removals:
        print(("removed" if args.apply else "would_remove") + f": {path}")

    if args.apply:
        for path in removals:
            path.unlink(missing_ok=True)
    else:
        print(f"\n{len(removals)} file(s) would be removed. Re-run with --apply to delete them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
