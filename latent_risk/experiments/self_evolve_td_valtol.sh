#!/usr/bin/env bash
# Single-judge self-evolving (payload-blind workflow) with a looser validation gate: a candidate
# may catch up to VALIDATION_MAX_LOST fewer verified dangerous validation messages than the
# incumbent if its validation precision rises by at least MIN_GAIN (0.01). In the task-deviation
# run (truncated/self_evolve_td) this gate accepts v3b (6/8 caught at 25% vs v1's 7/8 at 19%).
#
# prepare copies that run (replays excluded: labels are read from all of truncated/, so a copied
# replay folder would count twice), makes v3b the incumbent and drops the later iterations, which
# were built on v1 (regate.py). run continues the loop from v3b with the same settings.
#
#   latent_risk/experiments/self_evolve_td_valtol.sh prepare
#   latent_risk/experiments/self_evolve_td_valtol.sh run           # iterations 3..MAX_ITERATIONS
#   latent_risk/experiments/self_evolve_td_valtol.sh status
#   latent_risk/experiments/self_evolve_td_valtol.sh holdout-cost  # test pairs: attribute, print replay cost
#   latent_risk/experiments/self_evolve_td_valtol.sh holdout [K/N] # test pairs: replay-label flags + report
#
# Environment (defaults in brackets): STATE [truncated/self_evolve_td_vt]
#   FROM [truncated/self_evolve_td]  VALIDATION_MAX_LOST [1]  MAX_ITERATIONS [8]  PATIENCE [3]
#   and the settings of self_evolve_task_deviation.sh (PRECISION_TOLERANCE [0.02], LABEL_BUDGET [10], ...)
set -euo pipefail
cd "$(dirname "$0")/../.."

export STATE="${STATE:-truncated/self_evolve_td_vt}"
export VALIDATION_MAX_LOST="${VALIDATION_MAX_LOST:-1}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-8}"
FROM="${FROM:-truncated/self_evolve_td}"

case "${1:-}" in
  prepare)
    if [ -e "$STATE" ]; then
      echo "$STATE exists; remove it or set STATE to prepare again" >&2
      exit 1
    fi
    rsync -a --exclude /replays "$FROM/" "$STATE/"
    uv run python latent_risk/experiments/regate.py --state "$STATE" \
      --validation-max-lost "$VALIDATION_MAX_LOST" \
      --precision-tolerance "${PRECISION_TOLERANCE:-0.02}"
    ;;
  run|status|holdout-cost|holdout)
    latent_risk/experiments/self_evolve_task_deviation.sh "$@"
    ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
