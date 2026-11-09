#!/usr/bin/env bash
# What should the warning writer see? (see warning_views.py): the prefix up to the first
# deviation vs the complete trajectory vs a shorter prefix, on the flagged end-to-end firings;
# a judge checks each warning against the attacker's real goal. LLM calls only, no replays.
#
#   latent_risk/experiments/warning_views.sh run      # writers + judge, then the report
#   latent_risk/experiments/warning_views.sh report   # $OUT/report.md
#
# Environment (defaults in brackets): CELLS [gpt:skill_only gpt:standard]  MODEL [gpt-5.6-luna]
#   CONCURRENCY [8]  OUT [truncated/warning_views]  ENV_FILE [.env]
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
OUT="${OUT:-truncated/warning_views}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--cells ${CELLS:-gpt:skill_only gpt:standard} --model "${MODEL:-gpt-5.6-luna}"
  --concurrency "${CONCURRENCY:-8}" --out "$OUT")
PY=("${UV[@]}" python latent_risk/experiments/warning_views.py)
case "${1:-}" in
  run) mkdir -p "$OUT"; "${PY[@]}" run "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
