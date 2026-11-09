#!/usr/bin/env bash
# Look-ahead probe at first deviations, offline (see lookahead.py): cut after each labelled first
# deviation that the risk-score judge keeps (fd_risk_score.sh), let the agent continue for a few
# steps only, ROLLOUTS times, and check the first LOOKAHEAD new agent messages with the
# payload-blind task-deviation extractor. Compares the look-ahead with the judge's text score.
#
#   latent_risk/experiments/lookahead.sh plan     # cuts (no model calls)
#   latent_risk/experiments/lookahead.sh run      # short continuations (Docker, agent model)
#   latent_risk/experiments/lookahead.sh score    # extractor calls on the continuations
#   latent_risk/experiments/lookahead.sh report   # $OUT/report.md
#
# Environment (defaults in brackets): DATASETS [d1 gpt:skill_only gpt:standard]  MIN_SCORE [10]
#   ROLLOUTS [3]  LOOKAHEAD [3]  MAX_STEPS [10]  MODEL [gpt-5.6-luna]  CONCURRENCY [8]
#   REPLAY_CONCURRENCY [5]  SCORES [truncated/fd_risk_score]  OUT [truncated/lookahead]
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
OUT="${OUT:-truncated/lookahead}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--datasets ${DATASETS:-d1 gpt:skill_only gpt:standard} --min-score "${MIN_SCORE:-10}"
  --rollouts "${ROLLOUTS:-3}" --lookahead "${LOOKAHEAD:-3}" --max-steps "${MAX_STEPS:-10}"
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}"
  --replay-concurrency "${REPLAY_CONCURRENCY:-5}" --scores "${SCORES:-truncated/fd_risk_score}"
  --out "$OUT")
PY=("${UV[@]}" python latent_risk/experiments/lookahead.py)
case "${1:-}" in
  plan|report) "${PY[@]}" "$1" "${ARGS[@]}" ;;
  run|score) mkdir -p "$OUT"; "${PY[@]}" "$1" "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
