#!/usr/bin/env bash
# Small feasibility test of two warning ablations on end-to-end runs (see warning_ablation.py):
# timing (the inferred warning just before the attack action vs at the firing) and pointer (a
# writer not told that or where the agent deviated) and writer views (at the firing, warnings
# written from the complete trajectory or a shorter prefix; texts from warning_views.sh).
#
#   latent_risk/experiments/warning_ablation.sh select     # pick the runs
#   latent_risk/experiments/warning_ablation.sh generate   # warning texts (LLM)
#   latent_risk/experiments/warning_ablation.sh plan       # checkpoints (no model calls)
#   latent_risk/experiments/warning_ablation.sh run        # replays (Docker; the user starts it)
#   latent_risk/experiments/warning_ablation.sh report     # $OUT/report.md
#
# Environment (defaults in brackets): CELLS [gpt:skill_only gpt:standard]  PER_CELL [2]
#   MIN_BETWEEN [2]  SEED [0]  REPLAYS [5]  REPLAY_CONCURRENCY [5]  MODEL [gpt-5.6-luna]
#   OUT [truncated/warning_ablation]  STATE [truncated/inserted_context/warning_ablation]  ENV_FILE [.env]
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
OUT="${OUT:-truncated/warning_ablation}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--cells ${CELLS:-gpt:skill_only gpt:standard} --per-cell "${PER_CELL:-2}"
  --min-between "${MIN_BETWEEN:-2}" --seed "${SEED:-0}" --replays "${REPLAYS:-5}"
  --replay-concurrency "${REPLAY_CONCURRENCY:-5}" --model "${MODEL:-gpt-5.6-luna}" --out "$OUT"
  --state "${STATE:-truncated/inserted_context/warning_ablation}")
PY=("${UV[@]}" python latent_risk/experiments/warning_ablation.py)
case "${1:-}" in
  select|generate|plan|report) "${PY[@]}" "$1" "${ARGS[@]}" ;;
  run) mkdir -p "$OUT"; "${PY[@]}" run "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
