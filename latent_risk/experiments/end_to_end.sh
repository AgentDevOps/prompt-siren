#!/usr/bin/env bash
# End-to-end test of the runtime pipeline (see end_to_end.py): runtime extractor → stage 1 risk
# score → sampled next messages → combined judge → inferred warning, on every run of the dataset
# v2 gpt cells; flagged firings replayed with and without the warning (same time); attack success
# and benign task score with vs without the pipeline.
#
#   latent_risk/experiments/end_to_end.sh monitor   # runtime extractor calls (LLM)
#   latent_risk/experiments/end_to_end.sh decide    # judges, samples, warning texts (LLM)
#   latent_risk/experiments/end_to_end.sh plan      # checkpoints (no model calls)
#   latent_risk/experiments/end_to_end.sh run       # replays (Docker, agent model; the user starts it)
#   latent_risk/experiments/end_to_end.sh report    # $OUT/report.md
#
# Environment (defaults in brackets): CELLS [gpt:skill_only gpt:standard]  STAGE1 [10]
#   SAMPLES [5]  THRESHOLD [3]  REPLAYS [5]  MODEL [gpt-5.6-luna]  CONCURRENCY [8]
#   REPLAY_CONCURRENCY [5]  OUT [truncated/end_to_end]  STATE [truncated/inserted_context/end_to_end]
#   ENV_FILE [.env]  MANIFEST [rq3/dataset_v2/manifest.json]
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
OUT="${OUT:-truncated/end_to_end}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--manifest "${MANIFEST:-rq3/dataset_v2/manifest.json}" --cells ${CELLS:-gpt:skill_only gpt:standard} --stage1 "${STAGE1:-10}"
  --samples "${SAMPLES:-5}" --threshold "${THRESHOLD:-3}" --replays "${REPLAYS:-5}"
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}"
  --replay-concurrency "${REPLAY_CONCURRENCY:-5}" --out "$OUT"
  --state "${STATE:-truncated/inserted_context/end_to_end}")
PY=("${UV[@]}" python latent_risk/experiments/end_to_end.py)
case "${1:-}" in
  plan|report) "${PY[@]}" "$1" "${ARGS[@]}" ;;
  monitor|decide|run) mkdir -p "$OUT"; "${PY[@]}" "$1" "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
