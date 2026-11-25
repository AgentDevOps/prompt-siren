#!/usr/bin/env bash
# Label stability (paper plan step 4; see label_stability.py): for every screening label with 10
# replays, the first 5 vs replays 6-10 against the pair's fresh-run baseline. Reads only.
#
#   latent_risk/experiments/label_stability.sh    # $OUT
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest_full.json]
#   CELLS [gpt:skill_only gpt:standard qwen:skill_only qwen:standard]
#   OUT [truncated/label_stability/report.md]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="latent_risk/experiments:latent_risk${PYTHONPATH:+:$PYTHONPATH}"
command -v uv >/dev/null && PY=(uv run python) || PY=(.venv/bin/python)
# shellcheck disable=SC2206  # space-separated list
"${PY[@]}" latent_risk/experiments/label_stability.py --manifest "${MANIFEST:-rq3/dataset_v2/manifest_full.json}" \
  --cells ${CELLS:-gpt:skill_only gpt:standard qwen:skill_only qwen:standard} \
  --out "${OUT:-truncated/label_stability/report.md}"
