#!/usr/bin/env bash
# RQ3 pilot: generic risk warning at dangerous checkpoints (see latent_risk/experiments/intervention.py).
#
#   latent_risk/experiments/intervention.sh plan          # create the checkpoints
#   latent_risk/experiments/intervention.sh smoke         # 1 replay of one checkpoint, to check the mechanics
#   latent_risk/experiments/intervention.sh run [K/N]     # replay every checkpoint to REPLAYS runs
#   latent_risk/experiments/intervention.sh report        # truncated/intervention_pilot/report.md
#
# Environment (defaults in brackets): REPLAYS [10]  REPLAY_CONCURRENCY [3]  ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
PY=("${UV[@]}" python latent_risk/experiments/intervention.py)
case "${1:-}" in
  plan) "${PY[@]}" plan ;;
  smoke) "${PY[@]}" run --targets 277c4c2e:4 --conditions none-before --replays 1 ;;
  run)
    part=()
    [ -n "${2:-}" ] && part=(--part "$2")
    "${PY[@]}" run --replays "${REPLAYS:-10}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}" "${part[@]}"
    ;;
  report) "${PY[@]}" report ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
