#!/usr/bin/env bash
# Self-evolving attribution, round 6: round 5 plus a validation gate.
#
# Round 5 showed the remaining problem is overfitting: v4a raised training precision but lost the
# held-out positives. Round 6 adds a validation set: task pairs drawn at random (seeded) from the
# non-held-out ones, never trained on. A candidate that passes the training check is rejected if,
# on validation, it catches fewer verified dangerous messages or has a lower precision than the
# incumbent; the outcome (with excerpts of lost positives / new false alarms) is fed back to the
# optimiser. The held-out test task pairs stay untouched for the final result.
#
# Unchanged from round 5:
#   fix 1  the loop is scored on *every* failed training run, not only on runs that already had
#          a label; before iteration 1 it verifies v1's own flags by replay, afterwards each new
#          pick as usual (labels that already exist are reused, not replayed again)
#   fix 2  attack actions (harm signature) and later messages are not selectable candidates,
#          so they can't push the earlier, replay-relevant messages out of the top 3
#   fix 4  six task pairs are held out *before* evolving and never seen by the loop
#
#   latent_risk/experiments/self_evolve_round6.sh run              # verify v1's flags, then evolve from v1
#   latent_risk/experiments/self_evolve_round6.sh status
#   latent_risk/experiments/self_evolve_round6.sh holdout-cost     # held-out: attribute, print replay cost
#   latent_risk/experiments/self_evolve_round6.sh holdout [K/N]    # held-out: replay-label flags + report
#   latent_risk/experiments/self_evolve_round6.sh curve-cost       # learning curve: every accepted version
#   latent_risk/experiments/self_evolve_round6.sh curve [K/N]      #   on the held-out task pairs (overfitting
#                                                      #   check: does held-out follow training?)
#   latent_risk/experiments/self_evolve_round6.sh export           # write all replay labels into jobs/groups
#
# Results: $STATE/log.md (loop), ${STATE}_holdout/report.md (held-out).
#
# Environment (defaults in brackets):
#   STATE [truncated/self_evolve_r6]  MAX_ITERATIONS [18]  PATIENCE [8]
#   VALIDATION_TASKS [2]  VALIDATION_SEED [0]  random validation draw (stored in the state folder)
#   LABEL_BUDGET [10]  new picks replay-labelled per iteration
#   VERSIONS (holdout; default v1 and the final incumbent)
#   LABEL_CONCURRENCY [1]  (1 avoids parallel Docker image builds racing each other)
#   MIN_REPLAYS [5]  MAX_REPLAYS [10]  ZERO_SAFE_AT [10]  replays per labelled message
#     (development setting; use 5/30/10 for verification-grade labels). Every command uses the
#     same rule, so training and held-out labels stay comparable.
#   REPLAY_CONCURRENCY [3]  replays of one message run in parallel (same Docker image)
#   HOLDOUT_TASKS [the six task folders below]  ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."

STATE="${STATE:-truncated/self_evolve_r6}"
HOLDOUT_TASKS="${HOLDOUT_TASKS:-sympy_sympy-17139_skip_error_handling_task \
sympy_sympy-15345_modify_readme_curl_bash_task astropy_astropy-14182_skip_error_handling_task \
sympy_sympy-17139_multi_step_exfil_task astropy_astropy-14995_model_checkpoint_exfil_task \
sympy_sympy-15345_config_injection_exfil_task}"
LABEL_CONCURRENCY="${LABEL_CONCURRENCY:-1}"
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
EVOLVE=("${UV[@]}" --extra topics python latent_risk/scripts/self_evolve_attribution.py)
# shellcheck disable=SC2206  # space-separated list
COMMON=(--state "$STATE" --exclude-after-attack --eval-all-runs --holdout-tasks $HOLDOUT_TASKS
  --validation-tasks "${VALIDATION_TASKS:-2}" --validation-seed "${VALIDATION_SEED:-0}"
  --min-replays "${MIN_REPLAYS:-5}" --max-replays "${MAX_REPLAYS:-10}"
  --zero-safe-at "${ZERO_SAFE_AT:-10}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}")
part=()
[ -n "${2:-}" ] && part=(--part "$2")
versions=()
# shellcheck disable=SC2206
[ -n "${VERSIONS:-}" ] && versions=(--versions $VERSIONS)

accepted_versions() {
  # v1 and every version the loop accepted, in order (the learning curve)
  "${UV[@]}" python -c 'import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
print(" ".join(row["version"] for row in rows if row["accepted"]))' "$STATE/iterations.jsonl"
}

case "${1:-}" in
  run)
    "${EVOLVE[@]}" run "${COMMON[@]}" --verify-baseline \
      --label-budget "${LABEL_BUDGET:-10}" --label-concurrency "$LABEL_CONCURRENCY" \
      --max-iterations "${MAX_ITERATIONS:-18}" --patience "${PATIENCE:-8}"
    ;;
  status) "${EVOLVE[@]}" status "${COMMON[@]}" ;;
  holdout-cost) "${EVOLVE[@]}" holdout "${COMMON[@]}" "${versions[@]}" ;;
  holdout)
    "${EVOLVE[@]}" holdout "${COMMON[@]}" "${versions[@]}" --label \
      --label-concurrency "$LABEL_CONCURRENCY" "${part[@]}"
    ;;
  curve-cost)
    # shellcheck disable=SC2046
    "${EVOLVE[@]}" holdout "${COMMON[@]}" --versions $(accepted_versions)
    ;;
  curve)
    # shellcheck disable=SC2046
    "${EVOLVE[@]}" holdout "${COMMON[@]}" --versions $(accepted_versions) --label \
      --label-concurrency "$LABEL_CONCURRENCY" "${part[@]}"
    ;;
  export) "${UV[@]}" python latent_risk/scripts/self_evolve_attribution.py export-labels ;;
  *) sed -n '2,40p' "$0"; exit 1 ;;
esac
