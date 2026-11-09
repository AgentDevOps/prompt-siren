#!/usr/bin/env bash
# Runtime monitor check: does the task-deviation extractor, given only the user's task and the
# steps so far, fire at the run's offline first deviation? (see runtime_monitor.py). LLM calls
# only, no replays.
#
#   latent_risk/experiments/runtime_monitor.sh plan     # runs per group, extractor calls
#   latent_risk/experiments/runtime_monitor.sh run      # call the extractor step by step
#   latent_risk/experiments/runtime_monitor.sh report   # $OUT/report.md
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest.json]  AGENT [gpt]
#   CONDITION [skill_only]  SAFE [10]  CLEAN [5]  MODEL [gpt-5.6-luna]  CONCURRENCY [4]
#   MAX_REPLAYS [10; 5 for qwen, the study's label rule]  OUT [truncated/runtime_monitor_${AGENT}_$CONDITION]
#   SKIP_REJECTIONS [0] (1: a message that only declines the off-task work does not fire; report_turns.md)
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
AGENT="${AGENT:-gpt}" CONDITION="${CONDITION:-skill_only}"
[ "$AGENT" = qwen ] && MAX_REPLAYS="${MAX_REPLAYS:-5}"
OUT="${OUT:-truncated/runtime_monitor_${AGENT}_$CONDITION}"
ARGS=(--manifest "${MANIFEST:-rq3/dataset_v2/manifest.json}" --agent-model "$AGENT"
  --condition "$CONDITION" --safe "${SAFE:-10}" --clean "${CLEAN:-5}"
  --max-replays "${MAX_REPLAYS:-10}" --model "${MODEL:-gpt-5.6-luna}"
  --concurrency "${CONCURRENCY:-4}" --out "$OUT")
[ "${SKIP_REJECTIONS:-0}" = 1 ] && ARGS+=(--skip-rejections)
PY=("${UV[@]}" python latent_risk/experiments/runtime_monitor.py)
case "${1:-}" in
  plan|report) "${PY[@]}" "$1" "${ARGS[@]}" ;;
  run) "${PY[@]}" run "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
