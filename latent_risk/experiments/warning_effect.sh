#!/usr/bin/env bash
# Effect of the inferred warning at flagged firings (see warning_effect.py): baseline ASR, replayed
# ASR without and with the warning, and the benign score, per group (attack-succeeded runs;
# attack-failed runs by the firing point's replay label). Reads the end-to-end replays only.
#
#   latent_risk/experiments/warning_effect.sh    # truncated/warning_effect/report.md
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest.json]
#   E2E_OUT [truncated/end_to_end]  STATE [truncated/inserted_context/end_to_end]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="latent_risk/experiments:latent_risk${PYTHONPATH:+:$PYTHONPATH}"
command -v uv >/dev/null && PY=(uv run python) || PY=(.venv/bin/python)
"${PY[@]}" latent_risk/experiments/warning_effect.py --manifest "${MANIFEST:-rq3/dataset_v2/manifest.json}" \
  --e2e-out "${E2E_OUT:-truncated/end_to_end}" --state "${STATE:-truncated/inserted_context/end_to_end}"
