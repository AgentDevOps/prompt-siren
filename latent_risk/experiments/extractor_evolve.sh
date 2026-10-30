#!/usr/bin/env bash
# Payload-blind workflow: fixed judge, evolving chain extractor (see extractor_evolve.py).
#
# The judge is v1 of the task-deviation self-evolving run (truncated/self_evolve_td), frozen.
# Only the task-deviation extractor's selection criteria evolve; e1 is the original extractor
# (chains in rq3/task_deviation). Same splits, labels, training rule and validation gate as the
# single-judge run with VALIDATION_MAX_LOST=1 (self_evolve_td_valtol.sh).
#
#   latent_risk/experiments/extractor_evolve.sh run       # short run: MAX_ITERATIONS [4]
#   latent_risk/experiments/extractor_evolve.sh status
#   latent_risk/experiments/extractor_evolve.sh holdout-plan
#   latent_risk/experiments/extractor_evolve.sh holdout   # e1 vs final on the test pairs, <= MAX_LABEL replays
#
# Environment (defaults in brackets): OUT [truncated/extractor_evolve]  JUDGE [v1]
#   JUDGE_STATE [truncated/self_evolve_td]  MAX_ITERATIONS [4]  PATIENCE [3]  PROPOSALS [2]
#   LABEL_BUDGET [10]  VALIDATION_MAX_LOST [1]  MAX_LABEL [10]  CONCURRENCY [4]
#   REPLAY_CONCURRENCY [3]  LABEL_CONCURRENCY [1]  ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-truncated/extractor_evolve}"
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
RUN=("${UV[@]}" --extra topics python latent_risk/experiments/extractor_evolve.py
  --out "$OUT" --judge "${JUDGE:-v1}" --judge-state "${JUDGE_STATE:-truncated/self_evolve_td}"
  --reuse "truncated/self_evolve_td_vt_holdout/attributions/${JUDGE:-v1}"
  --max-iterations "${MAX_ITERATIONS:-4}" --patience "${PATIENCE:-3}"
  --proposals "${PROPOSALS:-2}" --label-budget "${LABEL_BUDGET:-10}"
  --validation-max-lost "${VALIDATION_MAX_LOST:-1}" --max-label "${MAX_LABEL:-10}"
  --concurrency "${CONCURRENCY:-4}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}"
  --label-concurrency "${LABEL_CONCURRENCY:-1}")
case "${1:-}" in
  run) mkdir -p "$OUT"; "${RUN[@]}" run 2>&1 | tee -a "$OUT.log" ;;
  status) "${RUN[@]}" status ;;
  holdout-plan) "${RUN[@]}" holdout --dry-run ;;
  holdout) "${RUN[@]}" holdout 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,18p' "$0"; exit 1 ;;
esac
