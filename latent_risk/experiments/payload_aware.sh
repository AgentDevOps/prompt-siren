#!/usr/bin/env bash
# Payload-aware upper bound for the runtime judges (see payload_aware.py): stage 1 and the
# combined judge with the attacker's goal added, vs the blind scores. LLM calls only.
#
#   latent_risk/experiments/payload_aware.sh score    # aware judges, then the report
#   latent_risk/experiments/payload_aware.sh report   # $OUT/report.md
#
# Environment (defaults in brackets): DATASETS [d1 gpt:skill_only gpt:standard]  MIN_SCORE [10]
#   MODEL [gpt-5.6-luna]  CONCURRENCY [8]  OUT [truncated/payload_aware]  ENV_FILE [.env]
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
OUT="${OUT:-truncated/payload_aware}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--datasets ${DATASETS:-d1 gpt:skill_only gpt:standard} --min-score "${MIN_SCORE:-10}"
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}" --out "$OUT")
PY=("${UV[@]}" python latent_risk/experiments/payload_aware.py)
case "${1:-}" in
  score) mkdir -p "$OUT"; "${PY[@]}" score "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
