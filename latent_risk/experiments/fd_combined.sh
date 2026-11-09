#!/usr/bin/env bash
# Judge + predictor at first deviations, frozen (see fd_combined.py): the risk-score judge's input
# plus the agent model's sampled next messages (self_predict.sh), with and without the
# extractor's flags; AUC vs the text judge and the two-stage pipeline. LLM calls only.
#
#   latent_risk/experiments/fd_combined.sh score    # score, then write $OUT/report.md
#   latent_risk/experiments/fd_combined.sh report   # $OUT/report.md
#
# Environment (defaults in brackets): DATASETS [d1 gpt:skill_only gpt:standard]
#   VARIANTS [samples samples+flags; also no-samples, lookahead, plan]  MIN_SCORE [10]  MODEL [gpt-5.6-luna]  CONCURRENCY [8]
#   SCORES [truncated/fd_risk_score]  SAMPLES [truncated/self_predict]  OUT [truncated/fd_combined]
#   ENV_FILE [.env]
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
OUT="${OUT:-truncated/fd_combined}"
# shellcheck disable=SC2206  # space-separated lists
ARGS=(--datasets ${DATASETS:-d1 gpt:skill_only gpt:standard}
  --variants ${VARIANTS:-samples samples+flags} --min-score "${MIN_SCORE:-10}"
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}"
  --scores "${SCORES:-truncated/fd_risk_score}" --samples "${SAMPLES:-truncated/self_predict}"
  --out "$OUT")
PY=("${UV[@]}" python latent_risk/experiments/fd_combined.py)
case "${1:-}" in
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  score) mkdir -p "$OUT"; "${PY[@]}" score "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
