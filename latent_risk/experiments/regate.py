"""Re-decide a copied self-evolving state under a looser validation gate (--validation-max-lost).

Walks the history in order and finds the first version that passed the training rule but was
rejected by the validation gate, and that the new gate accepts on its recorded validation scores
(e.g. v3b of the task-deviation run: 6/8 caught at 25% vs v1's 7/8 at 19%). That version becomes
the incumbent; every later iteration was built on the old incumbent, so it is dropped together
with its files (skills, attributions, validation, diffs, diagnosis). The re-decided version's
rules are recorded in provenance.json without cited cases (the iteration's case ids are gone).
settings.json gets the new gate setting, so `run` can continue the state with it.

    python latent_risk/experiments/regate.py --state truncated/self_evolve_td_vt --validation-max-lost 1
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

LATENT_RISK_ROOT = Path(__file__).resolve().parents[1]
if str(LATENT_RISK_ROOT) not in sys.path:
    sys.path.insert(0, str(LATENT_RISK_ROOT))

from scripts import self_evolve_attribution as se  # noqa: E402


def drop_files(state: Path, entry: dict) -> None:
    version = entry["version"]
    se.skill_path(state, version).unlink(missing_ok=True)
    (state / "diffs" / f"{version}.diff").unlink(missing_ok=True)
    (state / "diagnoses" / f"iteration{entry['iteration']}.json").unlink(missing_ok=True)
    for folder in ("attributions", "validation"):
        shutil.rmtree(state / folder / version, ignore_errors=True)


def regate(args: argparse.Namespace) -> None:
    state: Path = args.state
    entries = se.history(state)
    flipped = None
    for entry in entries:
        validation = entry.get("validation", {})
        parent = entry.get("parent")
        if entry["accepted"] or entry["version"] not in validation or parent not in validation:
            continue
        ok, summary = se.validation_gate(
            validation[entry["version"]],
            validation[parent],
            args.precision_tolerance,
            args.validation_max_lost,
            args.min_gain,
        )
        if ok:
            flipped = entry
            break
    if flipped is None:
        print("no rejected version passes the new gate; state unchanged")
        return
    kept = [entry for entry in entries if entry["iteration"] <= flipped["iteration"]]
    for entry in entries:
        if entry["iteration"] > flipped["iteration"]:
            drop_files(state, entry)
    flipped["accepted"] = True
    flipped["note"] = (
        f"{flipped['note']}; re-decided with --validation-max-lost {args.validation_max_lost}: "
        f"{summary}"
    )
    (state / "iterations.jsonl").write_text(
        "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in kept), encoding="utf-8"
    )
    parent_criteria = se.skill_path(state, flipped["parent"]).read_text(encoding="utf-8")
    se.record_provenance(
        state, flipped["parent"], flipped["version"], parent_criteria, flipped.get("plan", []), []
    )
    settings = se.load_json(state / "settings.json")
    settings["validation_max_lost"] = args.validation_max_lost
    (state / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    dropped = sorted({entry["version"] for entry in entries} - {entry["version"] for entry in kept})
    print(
        f"{flipped['version']} (iteration {flipped['iteration']}) is now the incumbent: {summary}. "
        f"Dropped later versions: {', '.join(dropped) or 'none'}"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--validation-max-lost", type=int, default=1)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--precision-tolerance", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    regate(parse_args(sys.argv[1:] if argv is None else argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
