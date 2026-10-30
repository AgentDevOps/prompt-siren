#!/usr/bin/env bash
# Self-evolving payload-blind judges with an OR ensemble: is OR still a good final rule once
# there are more than two judges?
#
# Continues the task-deviation run (truncated/self_evolve_td, iterations 0-3) in a copy, with
# --ensemble or. The judges take turns: each iteration revises one judge from its own errors,
# accepted by its own training rule and validation gate plus an ensemble guard (the OR keeps
# every verified dangerous validation message it catches). A revision that is not accepted joins
# as a new judge when the OR then catches more, at a precision cost of at most
# PRECISION_TOLERANCE. The earlier gate rejections (v3b, v4b) are considered first. PATIENCE
# counts rejections per judge. Then ensemble_eval.py compares, on
# training, validation and the held-out test pairs: each judge alone, OR of the first k judges,
# at-least-2 voting and the mean score; replays only for flags where OR and the others disagree.
#
#   latent_risk/experiments/self_evolve_td_ensemble.sh run         # copy the state once, evolve
#   latent_risk/experiments/self_evolve_td_ensemble.sh status
#   latent_risk/experiments/self_evolve_td_ensemble.sh attribute   # every member on the test runs (LLM only)
#   latent_risk/experiments/self_evolve_td_ensemble.sh label-plan  # disagreement flags to label, no replays
#   latent_risk/experiments/self_evolve_td_ensemble.sh label       # replay-label up to MAX_LABEL of them
#   latent_risk/experiments/self_evolve_td_ensemble.sh report      # truncated/ensemble_eval/report.md
#
# Environment (defaults in brackets): STATE [truncated/self_evolve_td_or]
#   FROM [truncated/self_evolve_td]  MAX_ITERATIONS [9]  PATIENCE [3]  LABEL_BUDGET [10]
#   OUT [truncated/ensemble_eval]  MAX_LABEL [10]  SEED [0]  CONCURRENCY [4]
#   REPLAY_CONCURRENCY [3]  LABEL_CONCURRENCY [1]  ENV_FILE [.env]
#   and the settings of self_evolve_task_deviation.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

export STATE="${STATE:-truncated/self_evolve_td_or}"
FROM="${FROM:-truncated/self_evolve_td}"
OUT="${OUT:-truncated/ensemble_eval}"
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
EVAL=("${UV[@]}" --extra topics python latent_risk/experiments/ensemble_eval.py
  --state "$STATE" --out "$OUT"
  --concurrency "${CONCURRENCY:-4}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}"
  --label-concurrency "${LABEL_CONCURRENCY:-1}" --max-label "${MAX_LABEL:-10}" --seed "${SEED:-0}")

case "${1:-}" in
  run)
    if [ ! -d "$STATE" ]; then
      # Replays stay in $FROM: labels are collected from all of truncated/, and a copied replay
      # folder would count twice.
      rsync -a --exclude /replays "$FROM/" "$STATE/"
    fi
    ENSEMBLE=or MAX_ITERATIONS="${MAX_ITERATIONS:-9}" \
      latent_risk/experiments/self_evolve_task_deviation.sh run
    ;;
  status) ENSEMBLE=or latent_risk/experiments/self_evolve_task_deviation.sh status ;;
  attribute) "${EVAL[@]}" attribute 2>&1 | tee -a "$OUT.log" ;;
  label-plan) "${EVAL[@]}" label --dry-run ;;
  label) "${EVAL[@]}" label 2>&1 | tee -a "$OUT.log" ;;
  report) "${EVAL[@]}" report ;;
  *) sed -n '2,27p' "$0"; exit 1 ;;
esac
