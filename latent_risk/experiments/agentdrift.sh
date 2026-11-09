#!/usr/bin/env bash
# The frozen runtime workflow on AgentDrift's step-labelled trajectories (see agentdrift.py):
# where the runtime extractor fires relative to their injection_point / first hijacked step, and
# how often the workflow flags resisted attacks, hard negatives and benign runs. LLM calls only.
#
#   latent_risk/experiments/agentdrift.sh monitor   # runtime extractor after every agent message
#   latent_risk/experiments/agentdrift.sh decide    # stage 1, sampled next steps, combined judge, warning
#   latent_risk/experiments/agentdrift.sh report    # $OUT/report.md
#
# Environment (defaults in brackets): DOMAINS [coding]  PER_CATEGORY [40]  SEED [0]  STAGE1 [10]
#   SAMPLES [5]  THRESHOLD [3]  MODEL [gpt-5.6-luna]  SAMPLER [meta-llama/llama-3.3-70b-instruct]
#   CONCURRENCY [8]  OUT [truncated/agentdrift]  ENV_FILE [.env]
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
OUT="${OUT:-truncated/agentdrift}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--domains ${DOMAINS:-coding} --per-category "${PER_CATEGORY:-40}" --seed "${SEED:-0}"
  --stage1 "${STAGE1:-10}" --samples "${SAMPLES:-5}" --threshold "${THRESHOLD:-3}"
  --model "${MODEL:-gpt-5.6-luna}" --sampler "${SAMPLER:-meta-llama/llama-3.3-70b-instruct}"
  --concurrency "${CONCURRENCY:-8}" --out "$OUT")
PY=("${UV[@]}" python latent_risk/experiments/agentdrift.py)
case "${1:-}" in
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  monitor|decide) mkdir -p "$OUT"; "${PY[@]}" "$1" "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
