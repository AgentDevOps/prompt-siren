#!/usr/bin/env bash
# AUC of the runtime workflow at its own firing points (see runtime_auc.py): the runtime
# extractor's firing step in each attack-failed run of the dataset v2 gpt cells is replay-labelled,
# then stage 1, the combined judge and the whole workflow are scored against those labels.
# Needs the runtime decisions first (end_to_end.sh monitor + decide with the same manifest):
#
#   MANIFEST=rq3/dataset_v2/manifest_swap.json OUT=truncated/end_to_end_swap \
#     STATE=truncated/inserted_context/end_to_end_swap latent_risk/experiments/end_to_end.sh monitor
#   (the same with decide)
#
#   latent_risk/experiments/runtime_auc.sh cost     # firing points still to label
#   latent_risk/experiments/runtime_auc.sh label    # replays (Docker; the user starts it)
#   latent_risk/experiments/runtime_auc.sh report   # $OUT/report.md
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest_swap.json]
#   RUNTIME [truncated/end_to_end_swap]  CELLS [gpt:skill_only gpt:standard]  STAGE1 [10]
#   MIN_REPLAYS [5]  MAX_REPLAYS [10]  REPLAY_CONCURRENCY [5]  LABEL_CONCURRENCY [4]
#   OUT [truncated/runtime_auc]  ENV_FILE [.env]
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
OUT="${OUT:-truncated/runtime_auc}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--manifest "${MANIFEST:-rq3/dataset_v2/manifest_swap.json}"
  --runtime "${RUNTIME:-truncated/end_to_end_swap}" --cells ${CELLS:-gpt:skill_only gpt:standard}
  --stage1 "${STAGE1:-10}" --min-replays "${MIN_REPLAYS:-5}" --max-replays "${MAX_REPLAYS:-10}"
  --replay-concurrency "${REPLAY_CONCURRENCY:-5}" --label-concurrency "${LABEL_CONCURRENCY:-4}"
  --out "$OUT")
PY=("${UV[@]}" python latent_risk/experiments/runtime_auc.py)
case "${1:-}" in
  cost|report) "${PY[@]}" "$1" "${ARGS[@]}" ;;
  label) mkdir -p "$OUT"; "${PY[@]}" label "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
