#!/usr/bin/env bash
# Self-evolving attribution for the payload-blind workflow (task-deviation chains + blind judge).
#
# Same loop and settings as round 11 (latent_risk/experiments/self_evolve_round11.sh), with one
# change: --workflow task-deviation. The judge ranks the task-deviation chains written by
# task_deviation_eval.sh run (rq3/task_deviation), v1 is the blind attribution criteria, and
# neither the judge nor the optimiser sees the attacker objective or payload. The replay labels
# (all of truncated/, including the task-deviation sample) and round 11's test and validation task
# pairs are reused, so the result compares directly with round 11. Exploration is off by default
# (EXPLORE_INITIAL=0, EXPLORE=0): the earlier rounds' explored labels are already in the pool.
# A short run by default (MAX_ITERATIONS=3).
#
#   latent_risk/experiments/self_evolve_task_deviation.sh run           # verify v1's flags, then evolve
#   latent_risk/experiments/self_evolve_task_deviation.sh status
#   latent_risk/experiments/self_evolve_task_deviation.sh holdout-cost  # held-out: attribute, print replay cost
#   latent_risk/experiments/self_evolve_task_deviation.sh holdout [K/N] # held-out: replay-label flags + report
#
# Results: $STATE/log.md (loop), ${STATE}_holdout/report.md (held-out).
#
# Environment (defaults in brackets):
#   STATE [truncated/self_evolve_td]  BLIND_DIR [rq3/task_deviation]
#   MAX_ITERATIONS [3]  PATIENCE [3]  LABEL_BUDGET [10]  CONTEXT_AFTER [1]
#   EXPLORE_INITIAL [0]  EXPLORE [0]  EXPLORE_SEED [0]  PRECISION_TOLERANCE [0.02]
#   SPLIT_FROM [truncated/self_evolve_r11]  VALIDATION_TASKS [2]  VALIDATION_SEED [0]
#   MIN_REPLAYS [5]  MAX_REPLAYS [10]  ZERO_SAFE_AT [10]  REPLAY_CONCURRENCY [3]
#   LABEL_CONCURRENCY [1]  VERSIONS (holdout; default v1 and the final incumbent)  ENV_FILE [.env]
#   ENSEMBLE [none] (or: gate-rejected versions may join an OR ensemble; see
#   self_evolve_td_ensemble.sh)  VALIDATION_MAX_LOST [0] (see self_evolve_td_valtol.sh)
#   OPTIMIZER_WORKSPACE [off] WORKSPACE_CALLS [30] (see self_evolve_td_workspace.sh)
set -euo pipefail
cd "$(dirname "$0")/../.."

STATE="${STATE:-truncated/self_evolve_td}"
LABEL_CONCURRENCY="${LABEL_CONCURRENCY:-1}"
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
EVOLVE=("${UV[@]}" --extra topics python latent_risk/scripts/self_evolve_attribution.py)

SPLIT_FROM="${SPLIT_FROM:-truncated/self_evolve_r11}"
for split in test_split.json validation_split.json; do
  if [ ! -f "$STATE/$split" ]; then
    mkdir -p "$STATE"
    cp "$SPLIT_FROM/$split" "$STATE/$split"
  fi
done
HOLDOUT_TASKS="$("${UV[@]}" python -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["tasks"]))' "$STATE/test_split.json")"
# shellcheck disable=SC2206  # space-separated list
COMMON=(--state "$STATE" --workflow task-deviation --blind-dir "${BLIND_DIR:-rq3/task_deviation}"
  --exclude-after-attack --eval-all-runs --holdout-tasks $HOLDOUT_TASKS
  --validation-tasks "${VALIDATION_TASKS:-2}" --validation-seed "${VALIDATION_SEED:-0}"
  --protect behaviour --precision-tolerance "${PRECISION_TOLERANCE:-0.02}"
  --explore-initial "${EXPLORE_INITIAL:-0}" --explore "${EXPLORE:-0}"
  --explore-seed "${EXPLORE_SEED:-0}" --context-after "${CONTEXT_AFTER:-1}"
  --min-replays "${MIN_REPLAYS:-5}" --max-replays "${MAX_REPLAYS:-10}"
  --zero-safe-at "${ZERO_SAFE_AT:-10}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}")
[ -n "${ENSEMBLE:-}" ] && COMMON+=(--ensemble "$ENSEMBLE")
[ -n "${VALIDATION_MAX_LOST:-}" ] && COMMON+=(--validation-max-lost "$VALIDATION_MAX_LOST")
LOOP=()
[ -n "${OPTIMIZER_WORKSPACE:-}" ] && LOOP+=(--optimizer-workspace --workspace-calls "${WORKSPACE_CALLS:-30}")
part=()
[ -n "${2:-}" ] && part=(--part "$2")
versions=()
# shellcheck disable=SC2206
[ -n "${VERSIONS:-}" ] && versions=(--versions $VERSIONS)

case "${1:-}" in
  run)
    "${EVOLVE[@]}" run "${COMMON[@]}" --verify-baseline \
      --label-budget "${LABEL_BUDGET:-10}" --label-concurrency "$LABEL_CONCURRENCY" \
      --max-iterations "${MAX_ITERATIONS:-3}" --patience "${PATIENCE:-3}" "${LOOP[@]}" 2>&1 | tee -a "$STATE.log"
    ;;
  status) "${EVOLVE[@]}" status "${COMMON[@]}" ;;
  holdout-cost) "${EVOLVE[@]}" holdout "${COMMON[@]}" "${versions[@]}" ;;
  holdout)
    "${EVOLVE[@]}" holdout "${COMMON[@]}" "${versions[@]}" --label \
      --label-concurrency "$LABEL_CONCURRENCY" "${part[@]}"
    ;;
  *) sed -n '2,28p' "$0"; exit 1 ;;
esac
