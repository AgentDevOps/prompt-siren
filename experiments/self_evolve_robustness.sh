#!/usr/bin/env bash
# Robustness check: repeat round variants with a different validation draw (and a fresh optimiser
# path), keeping round 7's test task pairs, so the variants can be compared more than once.
#
#   experiments/self_evolve_robustness.sh run      # every mode in MODES: baseline, iterations, curve
#   experiments/self_evolve_robustness.sh report   # one table: the original runs and the repeats
#
# Modes (each its own state truncated/self_evolve_robust_s<SEED>_<mode>):
#   verbatim   = round 7 (rules supporting a correct pick are frozen)
#   behaviour  = round 8 (regression set instead of frozen rules)
#   tolerant   = round 9 (round 8 + recall gains may cost a little precision)
#   explore    = round 10 (round 9 + skill-independent random exploration)
#
# Environment (defaults in brackets):
#   SEED [1]  validation seed of the repeat     MODES ["verbatim behaviour"]
#   ITERATIONS [3]  evolution iterations       (plus the variables of the round scripts)
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-1}"
MODES="${MODES:-verbatim behaviour}"
ITERATIONS="${ITERATIONS:-3}"
BASE="truncated/self_evolve_r7"  # its test split and v1 attributions are shared by every repeat
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")

script_for() {
  case "$1" in
    verbatim) echo experiments/self_evolve_round7.sh ;;
    behaviour) echo experiments/self_evolve_round8.sh ;;
    tolerant) echo experiments/self_evolve_round9.sh ;;
    explore) echo experiments/self_evolve_round10.sh ;;
    *) echo "unknown mode $1" >&2; exit 1 ;;
  esac
}

prepare() {  # same test task pairs and the same v1 judgments as round 7; new validation draw
  local state="$1"
  mkdir -p "$state/attributions/v1" "$state/validation/v1"
  [ -f "$state/test_split.json" ] || cp "$BASE/test_split.json" "$state/test_split.json"
  # v1 attributions are per run, so both of round 7's folders can fill both roles
  for folder in attributions/v1 validation/v1; do
    cp -n "$BASE/$folder/"*.json "$state/attributions/v1/" 2>/dev/null || true
    cp -n "$BASE/$folder/"*.json "$state/validation/v1/" 2>/dev/null || true
  done
}

case "${1:-}" in
  run)
    for mode in $MODES; do
      state="truncated/self_evolve_robust_s${SEED}_${mode}"
      prepare "$state"
      script="$(script_for "$mode")"
      export STATE="$state" VALIDATION_SEED="$SEED" SPLIT_FROM=/nonexistent
      echo "=== $mode (seed $SEED): $state"
      MAX_ITERATIONS=0 "$script" run
      MAX_ITERATIONS="$ITERATIONS" "$script" run
      "$script" curve
    done
    ;;
  report)
    "${UV[@]}" python - truncated/self_evolve_r7 truncated/self_evolve_r8 truncated/self_evolve_r9 truncated/self_evolve_r10 truncated/self_evolve_r11 \
      truncated/self_evolve_robust_s*_* <<'PY'
import json, sys
from pathlib import Path

print("| State | Protection | Tolerance | Validation draw | Accepted | Final | Training precision | "
      "Training dangerous ranked | Validation caught | Test caught | Test precision |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for name in sys.argv[1:]:
    state = Path(name)
    if not (state / "iterations.jsonl").exists():
        continue
    settings = json.loads((state / "settings.json").read_text())
    rows = [json.loads(line) for line in (state / "iterations.jsonl").read_text().splitlines()]
    accepted = [row for row in rows if row["accepted"]]
    final = accepted[-1]
    metrics = final.get("metrics") or {}
    validation = final.get("validation") or metrics.get("validation") or {}
    caught = validation.get(final["version"], {})
    split = json.loads((state / "validation_split.json").read_text())
    test = {}
    summary = Path(f"{state}_holdout") / "summary.json"
    if summary.exists():
        test = next((row for row in json.loads(summary.read_text()) if row["version"] == final["version"]), {})
    known = sum(json.loads((state / "test_split.json").read_text())["known_dangerous"].values())
    precision = test.get("precision")
    print(
        f"| {state.name} | {settings.get('protect', 'verbatim')} | {settings.get('precision_tolerance', 0)} | "
        f"seed {split['seed']} | {', '.join(row['version'] for row in accepted[1:]) or '-'} | {final['version']} | "
        f"{metrics.get('pessimistic_precision', 0):.2f} | {metrics.get('known_dangerous_ranked')}/{metrics.get('known_dangerous')} | "
        f"{len(caught.get('caught', [])) if caught else '-'}/{caught.get('known_dangerous', '-') if caught else '-'} | "
        f"{test.get('tp', '-')}/{known} | {'-' if precision is None else f'{precision:.2f}'} |"
    )
PY
    ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
