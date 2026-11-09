#!/usr/bin/env bash
# Agent-model predictor at first deviations, no tools (see self_predict.py): sample the agent
# model's next message SAMPLES times from the prefix up to each first deviation (the run's own
# prompt format) and check each with the payload-blind extractor; compare with the risk-score
# judge and the look-ahead probe. LLM calls only, no Docker. Needs fd_risk_score.sh and
# lookahead.sh plan (for the run configs).
#
#   latent_risk/experiments/self_predict.sh run      # sample and check, then $OUT/report.md
#   latent_risk/experiments/self_predict.sh report   # $OUT/report.md
#
# Environment (defaults in brackets): DATASETS [d1 gpt:skill_only gpt:standard]  MIN_SCORE [10]
#   SAMPLES [5]  MODEL [gpt-5.6-luna]  CONCURRENCY [8]  LOOKAHEAD [truncated/lookahead]
#   OUT [truncated/self_predict]  ENV_FILE [.env]
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
OUT="${OUT:-truncated/self_predict}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--datasets ${DATASETS:-d1 gpt:skill_only gpt:standard} --min-score "${MIN_SCORE:-10}"
  --samples "${SAMPLES:-5}" --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}"
  --lookahead "${LOOKAHEAD:-truncated/lookahead}" --out "$OUT")
PY=("${UV[@]}" python latent_risk/experiments/self_predict.py)
case "${1:-}" in
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  run) mkdir -p "$OUT"; "${PY[@]}" run "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
