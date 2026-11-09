#!/usr/bin/env bash
# Risk scores for first deviations (see fd_risk_score.py): a payload-blind judge, told that most
# flagged first deviations are false positives, scores each labelled first deviation 0-100; the
# report compares the scores with the replay verdicts (AUC, filtering at thresholds) and with v1.
# LLM calls only, no replays.
#
#   latent_risk/experiments/fd_risk_score.sh score    # score, then write $OUT/report.md
#   latent_risk/experiments/fd_risk_score.sh report   # $OUT/report.md from the saved scores
#
# Environment (defaults in brackets): DATASETS [d1 gpt:skill_only gpt:standard qwen:standard]
#   MODEL [gpt-5.6-luna]  CONCURRENCY [8]  OUT [truncated/fd_risk_score]  ENV_FILE [.env]
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
OUT="${OUT:-truncated/fd_risk_score}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--datasets ${DATASETS:-d1 gpt:skill_only gpt:standard qwen:standard}
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}" --out "$OUT")
PY=("${UV[@]}" python latent_risk/experiments/fd_risk_score.py)
case "${1:-}" in
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  score) mkdir -p "$OUT"; "${PY[@]}" score "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
