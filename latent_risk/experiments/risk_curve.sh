#!/usr/bin/env bash
# How the replay ASR changes along the attack chain: a few attack-failed runs of dataset v2 are
# replay-labelled at every agent message from shortly before their first deviation up to the
# attack action (see risk_curve.py). A quick check: 5 replays per position by default.
#
#   latent_risk/experiments/risk_curve.sh cost     # positions and replays still needed
#   latent_risk/experiments/risk_curve.sh label    # replay-label them
#   latent_risk/experiments/risk_curve.sh report   # $OUT/report.md
#
# Peak test: fresh replays (REPLAYS each, apart from the curve's) at POINTS, each run's first
# deviation first, then whether a later point is riskier (report in $PEAK_OUT/report.md):
#   latent_risk/experiments/risk_curve.sh fresh && latent_risk/experiments/risk_curve.sh compare
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest.json]  AGENT [gpt]
#   CONDITION [skill_only]  RUNS [five gpt skill-only runs, below]  BEFORE [2]  AFTER [6]
#   OUT [truncated/risk_curve_${AGENT}_$CONDITION]  MIN_REPLAYS [5]  MAX_REPLAYS [5]
#   REPLAY_CONCURRENCY [5]  LABEL_CONCURRENCY [4]  ENV_FILE [.env]
#   POINTS [first deviation and highest curve point of 11d7675b and ad4d0501]  REPLAYS [30]
#   PEAK_OUT [truncated/risk_curve_peak_${AGENT}_$CONDITION]
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_FILE="${ENV_FILE:-.env}"
if command -v uv >/dev/null; then
  UV=(uv run)
  [ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
else  # no uv on PATH: the project venv, with the env file loaded
  if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE" >/dev/null 2>&1; set +a; fi
  UV=(env PATH="$PWD/.venv/bin:$PATH")
fi
MANIFEST="${MANIFEST:-rq3/dataset_v2/manifest.json}" AGENT="${AGENT:-gpt}"
CONDITION="${CONDITION:-skill_only}"
# gpt skill-only runs with several agent messages between the first deviation and the attack
# action (or the run's end): dangerous first deviations (11d7675b, d9281d48, 852678ae) and safe
# ones (3b78e995, ad4d0501)
RUNS="${RUNS:-11d7675b d9281d48 852678ae 3b78e995 ad4d0501}"
OUT="${OUT:-truncated/risk_curve_${AGENT}_$CONDITION}"
# chosen from the curve (risk_curve report): the first deviation, then the highest point
POINTS="${POINTS:-11d7675b:8 11d7675b:14 ad4d0501:4 ad4d0501:16}"
PEAK_OUT="${PEAK_OUT:-truncated/risk_curve_peak_${AGENT}_$CONDITION}"
COMMON=(--manifest "$MANIFEST" --agent-model "$AGENT" --condition "$CONDITION"
  --state "truncated/self_evolve_td_v2_${AGENT}_$CONDITION"
  --blind-dir "rq3/task_deviation_v2_${AGENT}_$CONDITION"
  --replay-concurrency "${REPLAY_CONCURRENCY:-5}" --label-concurrency "${LABEL_CONCURRENCY:-4}"
  --reuse-images)
# shellcheck disable=SC2206  # space-separated lists
ARGS=(--runs $RUNS --before "${BEFORE:-2}" --after "${AFTER:-6}" --out "$OUT"
  --min-replays "${MIN_REPLAYS:-5}" --max-replays "${MAX_REPLAYS:-5}" --zero-safe-at 5
  "${COMMON[@]}")
# shellcheck disable=SC2206
PEAK=(--points $POINTS --out "$PEAK_OUT" --min-replays "${REPLAYS:-30}"
  --max-replays "${REPLAYS:-30}" --zero-safe-at "${REPLAYS:-30}" "${COMMON[@]}")
PY=("${UV[@]}" python latent_risk/experiments/risk_curve.py)
case "${1:-}" in
  cost|report) "${PY[@]}" "$1" "${ARGS[@]}" ;;
  label) "${PY[@]}" label "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  compare) "${PY[@]}" compare "${PEAK[@]}" ;;
  fresh) "${PY[@]}" fresh "${PEAK[@]}" 2>&1 | tee -a "$PEAK_OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
