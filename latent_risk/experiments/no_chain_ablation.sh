#!/usr/bin/env bash
# Ablation: attribution on the whole trajectory, without attack-chain extraction.
#
# Same as round 7's baseline (v1, gpt-6-luna judge, risk scores, attack-action cut-off, all failed
# training runs, v1's flags verified by replay, round 7's training / validation / test task
# pairs), except that the judge gets the whole trajectory as evidence and every agent message
# before the attack action as a candidate (--evidence trajectory). Only the attack payload context
# is kept; the chain judge's selection and summary are not used. No evolving.
#
#   latent_risk/experiments/no_chain_ablation.sh run       # v1 on training + validation, verify its new flags
#   latent_risk/experiments/no_chain_ablation.sh test      # v1 on the test task pairs (threshold from training)
#   latent_risk/experiments/no_chain_ablation.sh report    # whole trajectory vs attack chain (round 7)
#
# Environment (defaults in brackets): STATE [truncated/ablation_no_chain]
#   REPLAY_CONCURRENCY [3]  LABEL_CONCURRENCY [1]  ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."

STATE="${STATE:-truncated/ablation_no_chain}"
CHAIN_STATE="truncated/self_evolve_r7"   # the same setting with the attack chain
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
mkdir -p "$STATE"
for split in test_split.json validation_split.json; do
  [ -f "$STATE/$split" ] || cp "$CHAIN_STATE/$split" "$STATE/$split"
done
HOLDOUT_TASKS="$("${UV[@]}" python -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["tasks"]))' "$STATE/test_split.json")"
# shellcheck disable=SC2206
COMMON=(--state "$STATE" --evidence trajectory --exclude-after-attack --eval-all-runs
  --holdout-tasks $HOLDOUT_TASKS --validation-tasks 2 --validation-seed 0
  --min-replays 5 --max-replays 10 --zero-safe-at 10
  --replay-concurrency "${REPLAY_CONCURRENCY:-3}" --label-concurrency "${LABEL_CONCURRENCY:-1}")
EVOLVE=("${UV[@]}" --extra topics python latent_risk/scripts/self_evolve_attribution.py)

case "${1:-}" in
  run) "${EVOLVE[@]}" run "${COMMON[@]}" --verify-baseline --label-budget 10 --max-iterations 0 ;;
  test) "${EVOLVE[@]}" holdout "${COMMON[@]}" --versions v1 --label ;;
  report)
    "${UV[@]}" python - "$CHAIN_STATE" "$STATE" <<'PY'
import json, sys
from pathlib import Path

print("| Judge evidence | Threshold | Training flags | Training TP / FP | Training precision | Training dangerous caught | Validation caught (precision) | Test flags | Test TP / FP | Test caught |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for name, label in zip(sys.argv[1:], ("attack chain (round 7)", "whole trajectory")):
    state = Path(name)
    if not (state / "iterations.jsonl").exists():
        print(f"| {label} | (not run yet) |")
        continue
    v1 = json.loads((state / "iterations.jsonl").read_text().splitlines()[0])["metrics"]
    val = (v1.get("validation") or {}).get("v1", {})
    summary = Path(f"{state}_holdout") / "summary.json"
    test = next((row for row in json.loads(summary.read_text()) if row["version"] == "v1"), {}) if summary.exists() else {}
    known = sum(json.loads((state / "test_split.json").read_text())["known_dangerous"].values())
    vp = val.get("precision")
    print(
        f"| {label} | {v1['threshold']} | {v1['flagged']} | {v1['tp']} / {v1['fp']} | {v1['pessimistic_precision']:.2f} | "
        f"{v1['known_dangerous_ranked']}/{v1['known_dangerous']} | "
        f"{len(val.get('caught', []))}/{val.get('known_dangerous', '-')} ({'-' if vp is None else f'{vp:.2f}'}) | "
        f"{test.get('flagged', '-')} | {test.get('tp', '-')} / {test.get('fp', '-')} | {test.get('tp', '-')}/{known} |"
    )
PY
    ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
