#!/usr/bin/env bash
# Dangerous messages in dataset v2 per model x condition x task pair (see v2_dangerous.py): count,
# baseline ASR and replayed ASR. Reads labels only.
#
#   latent_risk/experiments/v2_dangerous.sh    # truncated/v2_dangerous/report.md
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest_swap.json]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="latent_risk/experiments:latent_risk${PYTHONPATH:+:$PYTHONPATH}"
command -v uv >/dev/null && PY=(uv run python) || PY=(.venv/bin/python)
"${PY[@]}" latent_risk/experiments/v2_dangerous.py --manifest "${MANIFEST:-rq3/dataset_v2/manifest_swap.json}"
