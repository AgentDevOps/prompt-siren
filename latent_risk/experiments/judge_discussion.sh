#!/usr/bin/env bash
# Two payload-blind judges (v1 and v3b of the task-deviation self-evolving run): final output by
# OR vs by one discussion round (see judge_discussion.py). Evaluated on training (thresholds),
# validation and the held-out test task pairs of truncated/self_evolve_td.
#
#   latent_risk/experiments/judge_discussion.sh attribute    # initial judges on test + discussion round (LLM only)
#   latent_risk/experiments/judge_discussion.sh label-plan   # disagreement flags to label, no replays
#   latent_risk/experiments/judge_discussion.sh label        # replay-label up to MAX_LABEL of them
#   latent_risk/experiments/judge_discussion.sh report       # truncated/judge_discussion/report.md
#
# Environment (defaults in brackets): CONCURRENCY [4]  REPLAY_CONCURRENCY [3]
#   LABEL_CONCURRENCY [1]  MAX_LABEL [10]  SEED [0]  ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
RUN=("${UV[@]}" --extra topics python latent_risk/experiments/judge_discussion.py
  --concurrency "${CONCURRENCY:-4}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}"
  --label-concurrency "${LABEL_CONCURRENCY:-1}" --max-label "${MAX_LABEL:-10}" --seed "${SEED:-0}")
case "${1:-}" in
  attribute) "${RUN[@]}" attribute 2>&1 | tee -a truncated/judge_discussion.log ;;
  label-plan) "${RUN[@]}" label --dry-run ;;
  label) "${RUN[@]}" label 2>&1 | tee -a truncated/judge_discussion.log ;;
  report) "${RUN[@]}" report ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
