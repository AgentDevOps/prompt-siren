#!/usr/bin/env bash
# Held-out check of evolved attribution skills on the attack-failed runs the loop never saw.
#
#   latent_risk/experiments/self_evolve_holdout.sh attribute  # attribute held-out runs, print the replay cost
#   latent_risk/experiments/self_evolve_holdout.sh label      # replay-label the flagged messages + report
#   latent_risk/experiments/self_evolve_holdout.sh label 1/3  # only the 1st of 3 shares (also 2/3, 3/3)
#   latent_risk/experiments/self_evolve_holdout.sh export     # write all replay labels into jobs/groups
#
# Each version keeps the score threshold frozen from training. Results:
# truncated/self_evolve_holdout/report.md
#
# Environment (defaults in brackets):
#   VERSIONS ["v1 v10a v17a"]  SAMPLE []  (e.g. 150: random sample, fixed at the first call)
#   LABEL_CONCURRENCY [2]  ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."

VERSIONS="${VERSIONS:-v1 v10a v17a}"
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
# shellcheck disable=SC2206  # space-separated list
args=(holdout --versions $VERSIONS)
[ -n "${SAMPLE:-}" ] && args+=(--holdout-sample "$SAMPLE")

case "${1:-}" in
  attribute) "${UV[@]}" --extra topics python latent_risk/scripts/self_evolve_attribution.py "${args[@]}" ;;
  label)
    # optional second argument K/N: label only that share, e.g. `label 2/3`
    [ -n "${2:-}" ] && args+=(--part "$2")
    "${UV[@]}" --extra topics python latent_risk/scripts/self_evolve_attribution.py "${args[@]}" \
      --label --label-concurrency "${LABEL_CONCURRENCY:-2}"
    ;;
  export) "${UV[@]}" python latent_risk/scripts/self_evolve_attribution.py export-labels ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
