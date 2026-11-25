#!/usr/bin/env bash
# Census manifest of dataset v2 (see full_manifest.py): the same cells with all valid runs instead
# of the 10-run sample. Writes $OUTPUT; the input manifest is unchanged. No model calls.
#
#   latent_risk/experiments/full_manifest.sh
#
# Environment (defaults in brackets): INPUT [rq3/dataset_v2/manifest_swap.json]
#   OUTPUT [rq3/dataset_v2/manifest_full.json]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="latent_risk/experiments:latent_risk${PYTHONPATH:+:$PYTHONPATH}"
command -v uv >/dev/null && PY=(uv run python) || PY=(.venv/bin/python)
"${PY[@]}" latent_risk/experiments/full_manifest.py --input "${INPUT:-rq3/dataset_v2/manifest_swap.json}" \
  --output "${OUTPUT:-rq3/dataset_v2/manifest_full.json}"
