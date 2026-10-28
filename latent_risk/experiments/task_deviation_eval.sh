#!/usr/bin/env bash
# Payload-blind workflow (task-deviation extraction + blind attribution) vs the current workflow,
# scored on the replay labels of the 8 in-scope task pairs (see task_deviation_eval.py).
#
#   latent_risk/experiments/task_deviation_eval.sh smoke        # one run, to check the outputs
#   latent_risk/experiments/task_deviation_eval.sh run          # extraction + attribution, all runs
#   latent_risk/experiments/task_deviation_eval.sh report       # rq3/task_deviation/report.md
#   latent_risk/experiments/task_deviation_eval.sh label-plan   # unlabelled blind flags, no replays
#   latent_risk/experiments/task_deviation_eval.sh label        # replay-label them (5-10 replays)
#   latent_risk/experiments/task_deviation_eval.sh sample-plan  # PER_PAIR flags per task pair, no replays
#   latent_risk/experiments/task_deviation_eval.sh sample       # replay-label only that sample
#
# Environment (defaults in brackets): MODEL [gpt-5.6-luna]  CONCURRENCY [4]  TOP_K [3]
#   REPLAY_CONCURRENCY [3]  PER_PAIR [2]  SEED [0]  ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
OUT=rq3/task_deviation
EVAL=("${UV[@]}" python latent_risk/experiments/task_deviation_eval.py --blind-dir "$OUT" --top-k "${TOP_K:-3}")
BLIND=("${UV[@]}" python latent_risk/scripts/task_deviation_attribution.py --output-dir "$OUT"
  --model "${MODEL:-gpt-5.6-luna}" --top-k "${TOP_K:-3}" --concurrency "${CONCURRENCY:-4}")
mapfile -t JOBS < <("${EVAL[@]}" runs)
case "${1:-}" in
  smoke) "${BLIND[@]}" "${JOBS[@]}" --run-ids 277c4c2e ;;
  run) "${BLIND[@]}" "${JOBS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  report) "${EVAL[@]}" report ;;
  label-plan) "${EVAL[@]}" label --dry-run ;;
  label) "${EVAL[@]}" label --replay-concurrency "${REPLAY_CONCURRENCY:-3}" 2>&1 | tee -a "$OUT.label.log" ;;
  sample-plan) "${EVAL[@]}" label --per-pair "${PER_PAIR:-2}" --seed "${SEED:-0}" --dry-run ;;
  sample) "${EVAL[@]}" label --per-pair "${PER_PAIR:-2}" --seed "${SEED:-0}" \
    --replay-concurrency "${REPLAY_CONCURRENCY:-3}" 2>&1 | tee -a "$OUT.label.log" ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
