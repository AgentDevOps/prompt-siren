#!/usr/bin/env bash
# Usage-based swap of the dataset v2 sample (see swap_sample.py): the most-used D1 runs go into the
# gpt standard sample of the original 8 task pairs, unused sampled runs go back to D1; per pair the
# sample size and attack-succeeded count stay. Writes a new manifest; the current one is unchanged.
#
#   latent_risk/experiments/swap_sample.sh plan    # report only
#   latent_risk/experiments/swap_sample.sh write   # report + $OUTPUT
#
# Environment (defaults in brackets): OUTPUT [rq3/dataset_v2/manifest_swap.json]
#   PREFER [32456545 277c4c2e e033d6b8 5876d967: the D1 replay examples and pilot targets]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="latent_risk/experiments:latent_risk${PYTHONPATH:+:$PYTHONPATH}"
PY=python
command -v uv >/dev/null && PY="uv run python" || PY=".venv/bin/python"
# shellcheck disable=SC2206  # space-separated list
PREFER=(${PREFER:-32456545 277c4c2e e033d6b8 5876d967})
case "${1:-}" in
  plan|write) $PY latent_risk/experiments/swap_sample.py "$1" \
    --output "${OUTPUT:-rq3/dataset_v2/manifest_swap.json}" --prefer "${PREFER[@]}" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
