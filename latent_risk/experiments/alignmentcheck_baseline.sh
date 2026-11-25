#!/usr/bin/env bash
# LlamaFirewall's AlignmentCheck (prompt verbatim from PurpleLlama) as a baseline for the runtime
# extractor R1 (see alignmentcheck_baseline.py): run after every agent message of the dataset v2
# runs, then compared with R1 and IntentGuard's Plan Gate on the same runs. LLM calls only.
#
#   latent_risk/experiments/alignmentcheck_baseline.sh monitor   # AlignmentCheck calls (LLM)
#   latent_risk/experiments/alignmentcheck_baseline.sh report    # $OUT/report.md
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest_full.json]
#   CELLS [gpt:skill_only gpt:standard qwen:skill_only qwen:standard]
#   R1 [truncated/end_to_end_full]  INTENTGUARD [truncated/intentguard_baseline_full; none: leave out]
#   MODEL [gpt-5.6-luna]  CONCURRENCY [8]  OUT [truncated/alignmentcheck_baseline_full]
#   ENV_FILE [.env]  SCREEN [0; 1: labels with the screening rule of census_label.sh]
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_FILE="${ENV_FILE:-.env}"
if command -v uv >/dev/null; then
  UV=(uv run)
  [ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
else  # no uv on PATH: the project venv, with the env file loaded
  if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE" >/dev/null 2>&1; set +a; fi
  UV=(env PATH="$PWD/.venv/bin:$PATH")
fi
export PYTHONPATH="latent_risk/experiments:latent_risk${PYTHONPATH:+:$PYTHONPATH}"
OUT="${OUT:-truncated/alignmentcheck_baseline_full}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--manifest "${MANIFEST:-rq3/dataset_v2/manifest_full.json}"
  --cells ${CELLS:-gpt:skill_only gpt:standard qwen:skill_only qwen:standard}
  --r1 "${R1:-truncated/end_to_end_full}" --intentguard "${INTENTGUARD:-truncated/intentguard_baseline_full}"
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}" --out "$OUT")
[ "${SCREEN:-0}" = 1 ] && ARGS+=(--screen)
PY=("${UV[@]}" python latent_risk/experiments/alignmentcheck_baseline.py)
case "${1:-}" in
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  monitor) mkdir -p "$OUT"; "${PY[@]}" monitor "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
