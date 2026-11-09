#!/usr/bin/env bash
# Paper experiments E1.2 (prevalence of hidden high-risk states) and E2.1 (the frozen v1
# payload-blind workflow vs simple baselines), on the attack-failed runs of the 8 in-scope task
# pairs (see prevalence_baselines.py). Existing replay labels in truncated/ are reused; new
# replays go to $OUT/replays.
#
#   latent_risk/experiments/prevalence_baselines.sh cost          # selections, replays still needed
#   latent_risk/experiments/prevalence_baselines.sh label [K/N]   # replay-label them (optional share)
#   latent_risk/experiments/prevalence_baselines.sh report        # $OUT/report.md
#
# Labelling in stages (METHODS limits what is labelled; the report always shows every method):
#   METHODS="first-deviation v1-top1" latent_risk/experiments/prevalence_baselines.sh label
#   latent_risk/experiments/prevalence_baselines.sh label         # then first-reply and random
#
# Environment (defaults in brackets): OUT [truncated/prevalence_baselines]
#   STATE [truncated/self_evolve_td] (frozen v1, splits)  SAMPLE_PER_PAIR [6] (first-reply, random)
#   METHODS [all]  SEED [0]  MIN_REPLAYS [5]  MAX_REPLAYS [10]  ZERO_SAFE_AT [10]
#   REPLAY_CONCURRENCY [3]  LABEL_CONCURRENCY [1]  ENV_FILE [.env]
#   REUSE_IMAGES [0]: 1 makes replay jobs use the existing SWE-bench images instead of rebuilding
#   them. Only safe when the images on disk were built for the same injection condition: the image
#   tags do not name the condition, so skill-only and standard builds overwrite each other.
#
# Dataset v2 (rq3/framework_report.md 7.2): with MANIFEST, the population is the manifest's sampled
# attack-failed runs of AGENT [gpt] and CONDITION [standard]; first run dataset_v1_eval.sh prepare,
# extract and attribute with the same MANIFEST, AGENT and CONDITION. Defaults then become
# STATE truncated/self_evolve_td_v2_${AGENT}_$CONDITION, BLIND_DIR rq3/task_deviation_v2_${AGENT}_
# $CONDITION, OUT truncated/prevalence_v2_${AGENT}_$CONDITION and REUSE_IMAGES 1 (the images carry
# their condition in the tag and prepare_datasets.sh checks them; replays run next to other jobs).
#   MANIFEST=rq3/dataset_v2/manifest.json AGENT=gpt CONDITION=skill_only \
#     latent_risk/experiments/prevalence_baselines.sh cost
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
V2=()
if [ -n "${MANIFEST:-}" ]; then
  AGENT="${AGENT:-gpt}" CONDITION="${CONDITION:-standard}"
  STATE="${STATE:-truncated/self_evolve_td_v2_${AGENT}_$CONDITION}"
  OUT="${OUT:-truncated/prevalence_v2_${AGENT}_$CONDITION}"
  REUSE_IMAGES="${REUSE_IMAGES:-1}"
  V2=(--manifest "$MANIFEST" --agent-model "$AGENT" --condition "$CONDITION"
    --blind-dir "${BLIND_DIR:-rq3/task_deviation_v2_${AGENT}_$CONDITION}")
fi
ARGS=(--out "${OUT:-truncated/prevalence_baselines}" --state "${STATE:-truncated/self_evolve_td}"
  --sample-per-pair "${SAMPLE_PER_PAIR:-6}" --seed "${SEED:-0}"
  --min-replays "${MIN_REPLAYS:-5}" --max-replays "${MAX_REPLAYS:-10}"
  --zero-safe-at "${ZERO_SAFE_AT:-10}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}"
  --label-concurrency "${LABEL_CONCURRENCY:-1}" "${V2[@]}")
[ "${REUSE_IMAGES:-0}" = 1 ] && ARGS+=(--reuse-images)
# shellcheck disable=SC2206  # space-separated list
[ -n "${METHODS:-}" ] && ARGS+=(--methods $METHODS)
PY=("${UV[@]}" python latent_risk/experiments/prevalence_baselines.py)
case "${1:-}" in
  cost) "${PY[@]}" cost "${ARGS[@]}" ;;
  label)
    part=()
    [ -n "${2:-}" ] && part=(--part "$2")
    "${PY[@]}" label "${ARGS[@]}" "${part[@]}" 2>&1 | tee -a "${OUT:-truncated/prevalence_baselines}.log"
    ;;
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
