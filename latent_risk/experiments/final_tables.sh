#!/usr/bin/env bash
# Final tables for the paper (step 11; see final_tables.py). Reads only: no model calls, no
# replays. Method R1 -> R3 -> R4 -> R5; the first version (with the context-only stage 1 filter)
# alongside. Output: $OUT (default truncated/final_tables/report.md).
#
#   latent_risk/experiments/final_tables.sh
#
# Environment (defaults in brackets): CELLS [gpt:skill_only gpt:standard qwen:skill_only qwen:standard]
#   MANIFEST [rq3/dataset_v2/manifest_full.json]  OUT [truncated/final_tables/report.md]
#   SCREEN [1: labels with the screening rule]  BOOTSTRAP [2000]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="latent_risk/experiments:latent_risk${PYTHONPATH:+:$PYTHONPATH}"
command -v uv >/dev/null && PY=(uv run python) || PY=(.venv/bin/python)
# shellcheck disable=SC2206  # space-separated list
ARGS=(--manifest "${MANIFEST:-rq3/dataset_v2/manifest_full.json}"
  --cells ${CELLS:-gpt:skill_only gpt:standard qwen:skill_only qwen:standard}
  --out "${OUT:-truncated/final_tables/report.md}" --bootstrap "${BOOTSTRAP:-2000}"
  # development runs, as in held_out.sh
  --dev-manifests rq3/dataset_v2/manifest.json rq3/dataset_v2/manifest_swap.json
  --dev-cells gpt:skill_only gpt:standard qwen:standard --dev-jobs rq3/task_deviation)
[ "${SCREEN:-1}" = 1 ] && ARGS+=(--screen)
"${PY[@]}" latent_risk/experiments/final_tables.py "${ARGS[@]}"
