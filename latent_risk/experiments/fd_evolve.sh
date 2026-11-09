#!/usr/bin/env bash
# Evolve the combined judge's instructions as a held-out test (see fd_evolve.py): split by task
# pair, evolve on the training pairs from the judge's worst errors, test once on the held-out
# pairs (initial vs final instructions). LLM calls only; needs fd_risk_score.sh, self_predict.sh
# and lookahead.sh plan.
#
#   latent_risk/experiments/fd_evolve.sh evolve   # $OUT/summary.json, history.json
#
# Environment (defaults in brackets): DATASETS [d1 gpt:skill_only gpt:standard]  ROUNDS [4]
#   CASES [6]  SEED [0]  MIN_SCORE [10]  MODEL [gpt-5.6-luna]  CONCURRENCY [8]
#   OUT [truncated/fd_evolve]  ENV_FILE [.env]
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
OUT="${OUT:-truncated/fd_evolve}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--datasets ${DATASETS:-d1 gpt:skill_only gpt:standard} --rounds "${ROUNDS:-4}"
  --cases "${CASES:-6}" --seed "${SEED:-0}" --min-score "${MIN_SCORE:-10}"
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}" --out "$OUT")
case "${1:-}" in
  evolve) mkdir -p "$OUT"; "${UV[@]}" python latent_risk/experiments/fd_evolve.py evolve "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
