#!/usr/bin/env bash
# Does the optimiser propose better judge criteria when it can browse the training set's files?
# Same single-judge loop as self_evolve_td_valtol.sh (payload-blind workflow, validation gate
# with VALIDATION_MAX_LOST=1), continued from its incumbent v3b, with one change:
# --optimizer-workspace. The analysis step gets read-only tools over the training runs, the
# judge's scores and reasons, and the replays of every training label, and reports "findings"
# (observable features the replays suggest) to the proposal step. Without the workspace, the
# iterations from v3b (4-6 of truncated/self_evolve_td_vt) were all rejected.
#
# prepare copies truncated/self_evolve_td_vt (replays excluded: labels are read from all of
# truncated/, so a copied replay folder would count twice).
#
#   latent_risk/experiments/self_evolve_td_workspace.sh prepare
#   latent_risk/experiments/self_evolve_td_workspace.sh run       # iterations 6..MAX_ITERATIONS
#   latent_risk/experiments/self_evolve_td_workspace.sh status
#   latent_risk/experiments/self_evolve_td_workspace.sh holdout-cost | holdout [K/N]
#
# Findings and tool calls per iteration: $STATE/diagnoses/iteration<N>.json
#
# Environment (defaults in brackets): STATE [truncated/self_evolve_td_ws]
#   FROM [truncated/self_evolve_td_vt]  VALIDATION_MAX_LOST [1]  MAX_ITERATIONS [9]
#   WORKSPACE_CALLS [30]  and the settings of self_evolve_task_deviation.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

export STATE="${STATE:-truncated/self_evolve_td_ws}"
export VALIDATION_MAX_LOST="${VALIDATION_MAX_LOST:-1}"
export MAX_ITERATIONS="${MAX_ITERATIONS:-9}"
export OPTIMIZER_WORKSPACE=1
FROM="${FROM:-truncated/self_evolve_td_vt}"

case "${1:-}" in
  prepare)
    if [ -e "$STATE" ]; then
      echo "$STATE exists; remove it or set STATE to prepare again" >&2
      exit 1
    fi
    rsync -a --exclude /replays "$FROM/" "$STATE/"
    ;;
  run|status|holdout-cost|holdout)
    latent_risk/experiments/self_evolve_task_deviation.sh "$@"
    ;;
  *) sed -n '2,22p' "$0"; exit 1 ;;
esac
