#!/usr/bin/env bash
# IntentGuard's Plan Gate (intent-deviation check, prompt verbatim from the paper) as a baseline
# for the runtime extractor R1 (see intentguard_baseline.py): run after every agent message of the
# dataset v2 gpt runs, then compared with R1's firings on the same runs. LLM calls only.
#
#   latent_risk/experiments/intentguard_baseline.sh monitor   # gate calls (LLM)
#   latent_risk/experiments/intentguard_baseline.sh report    # $OUT/report.md
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest_swap.json]
#   CELLS [gpt:skill_only gpt:standard]  R1 [truncated/end_to_end_swap]  MODEL [gpt-5.6-luna]
#   MALICIOUS [0: 1 adds the malicious-intent check]  CONCURRENCY [8]
#   OUT [truncated/intentguard_baseline, or ..._malicious with MALICIOUS=1]  ENV_FILE [.env]
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
DEFAULT_OUT=truncated/intentguard_baseline
[ "${MALICIOUS:-0}" = 1 ] && DEFAULT_OUT=truncated/intentguard_baseline_malicious
OUT="${OUT:-$DEFAULT_OUT}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--manifest "${MANIFEST:-rq3/dataset_v2/manifest_swap.json}"
  --cells ${CELLS:-gpt:skill_only gpt:standard} --r1 "${R1:-truncated/end_to_end_swap}"
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}" --out "$OUT")
[ "${MALICIOUS:-0}" = 1 ] && ARGS+=(--malicious)
PY=("${UV[@]}" python latent_risk/experiments/intentguard_baseline.py)
case "${1:-}" in
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  monitor) mkdir -p "$OUT"; "${PY[@]}" monitor "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
