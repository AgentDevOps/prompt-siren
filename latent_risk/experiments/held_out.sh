#!/usr/bin/env bash
# Held-out view (paper plan step 2; see held_out.py): runtime AUC, the warning effect and the
# baselines recomputed on the development runs and on the held-out runs of each agent model.
# Reads existing outputs only (no model calls, no replays).
#
#   latent_risk/experiments/held_out.sh    # $OUT
#
# Development runs = what the runtime method (stage 1 threshold, next-step judge, warning) was
# built and tuned on: the D1 jobs (all 240 gpt standard runs of the 8 original task pairs, listed
# in rq3/task_deviation), the v2 gpt sample (manifest.json and the swapped manifest_swap.json, both
# conditions) and the v2 qwen standard sample (its first deviations were in the pool used for the
# stage 1 threshold). qwen skill-only was never used. Everything else in the census is held out.
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest_full.json]
#   OUT [truncated/held_out/report.md]  SCREEN [1: labels with the screening rule]
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="latent_risk/experiments:latent_risk${PYTHONPATH:+:$PYTHONPATH}"
command -v uv >/dev/null && PY=(uv run python) || PY=(.venv/bin/python)
ARGS=(--manifest "${MANIFEST:-rq3/dataset_v2/manifest_full.json}"
  --dev-manifests rq3/dataset_v2/manifest.json rq3/dataset_v2/manifest_swap.json
  --dev-cells gpt:skill_only gpt:standard qwen:standard
  --dev-jobs rq3/task_deviation
  --out "${OUT:-truncated/held_out/report.md}")
[ "${SCREEN:-1}" = 1 ] && ARGS+=(--screen)
"${PY[@]}" latent_risk/experiments/held_out.py "${ARGS[@]}"
