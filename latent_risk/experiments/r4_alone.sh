#!/usr/bin/env bash
# Ablation: R4 (sampled next messages + combined judge) on every attack-failed firing, without the
# stage 1 filter (see r4_alone.py). score samples the agent's own model (gpt: API; qwen: the local
# vLLM server, client requests only) and calls the judge; report compares with the deployed
# workflow, overall and on held_out.sh's development / held-out runs. The method without stage 1
# also warns at the firings only R4 alone flags: warn / plan / run / effect.
#
#   latent_risk/experiments/r4_alone.sh score    # R3 + R4 at the firings stage 1 dropped (LLM)
#   SUCCEEDED=1 latent_risk/experiments/r4_alone.sh score  # also attack-succeeded runs' firings
#   latent_risk/experiments/r4_alone.sh report   # $OUT/report.md
#   latent_risk/experiments/r4_alone.sh warn     # R5 warning texts at the newly flagged firings (LLM)
#   latent_risk/experiments/r4_alone.sh plan     # their checkpoints under $STATE (no model calls)
#   latent_risk/experiments/r4_alone.sh run      # replays without / with the warning (Docker; the user)
#   latent_risk/experiments/r4_alone.sh effect   # $OUT/warning_effect.md
#
# Environment (defaults in brackets): CELLS [gpt:skill_only gpt:standard]
#   MANIFEST [rq3/dataset_v2/manifest_full.json]  RUNTIME [truncated/end_to_end_full]
#   MODEL [gpt-5.6-luna]  CONCURRENCY [8]  OUT [truncated/r4_alone]  ENV_FILE [.env]
#   SCREEN [1: labels with the screening rule]  SUCCEEDED [0]  REPLAYS [3]  REPLAY_CONCURRENCY [5]
#   STATE [truncated/inserted_context/r4_alone]  E2E_STATE [truncated/inserted_context/end_to_end]
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
OUT="${OUT:-truncated/r4_alone}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--manifest "${MANIFEST:-rq3/dataset_v2/manifest_full.json}"
  --cells ${CELLS:-gpt:skill_only gpt:standard} --runtime "${RUNTIME:-truncated/end_to_end_full}"
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}" --out "$OUT"
  # development runs, as in held_out.sh
  --dev-manifests rq3/dataset_v2/manifest.json rq3/dataset_v2/manifest_swap.json
  --dev-cells gpt:skill_only gpt:standard qwen:standard --dev-jobs rq3/task_deviation
  --state "${STATE:-truncated/inserted_context/r4_alone}"
  --e2e-state "${E2E_STATE:-truncated/inserted_context/end_to_end}"
  --replays "${REPLAYS:-3}" --replay-concurrency "${REPLAY_CONCURRENCY:-5}")
[ "${SCREEN:-1}" = 1 ] && ARGS+=(--screen)
[ "${SUCCEEDED:-0}" = 1 ] && ARGS+=(--succeeded)
PY=("${UV[@]}" python latent_risk/experiments/r4_alone.py)
case "${1:-}" in
  report|plan|effect) "${PY[@]}" "$1" "${ARGS[@]}" ;;
  score|warn|run) mkdir -p "$OUT"; "${PY[@]}" "$1" "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
