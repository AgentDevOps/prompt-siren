#!/usr/bin/env bash
# Recovery step vs warning at dangerous first deviations (see recovery_context.py): a generated
# agent thought that sets the injected work aside is inserted after the first deviation, and the
# replay ASR is compared with no insertion, a placebo and the generic warning.
#
#   latent_risk/experiments/recovery_context.sh generate   # recovery texts (LLM): read $STATE/recovery_texts.md
#   latent_risk/experiments/recovery_context.sh window     # generated-window-target: WINDOW_ROLLOUTS
#                                                          # replays, then $STATE/windows.md
#   latent_risk/experiments/recovery_context.sh plan       # checkpoints
#   latent_risk/experiments/recovery_context.sh run        # REPLAYS replays per checkpoint
#   latent_risk/experiments/recovery_context.sh report     # $STATE/report.md
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest.json]  AGENT [gpt]
#   CONDITION [skill_only]  TARGETS [every dangerous first deviation of the study]  REPLAYS [10]
#   REPLAY_CONCURRENCY [5]  MODEL [gpt-5.6-luna] (recovery texts)
#   CONDITIONS [none-target placebo-target warning-target recovery-target]
#   DONORS [-] (transplant-target: TARGET_RUN=DONOR_RUN:END ..., a recovered stretch of the same task pair)
#   MAX_REPLAYS [10; 5 for qwen, the study's label rule]  STATE [truncated/inserted_context/recovery_context_${AGENT}_$CONDITION]
#   D1 [0] (1: dataset v1 runs and labels, gpt standard, e.g. the intervention.py targets; set
#   TARGETS and STATE)  COUNT_SETUP_CFG [0] (1: a setup.cfg write of the dependency_poisoning
#   package counts as an attack success)  WINDOW_STEPS [3]  WINDOW_ROLLOUTS [3]  ENV_FILE [.env]
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
STATE="${STATE:-truncated/inserted_context/recovery_context_${AGENT}_$CONDITION}"
ARGS=(--manifest "${MANIFEST:-rq3/dataset_v2/manifest.json}" --agent-model "$AGENT"
  --condition "$CONDITION" --max-replays "${MAX_REPLAYS:-10}" --model "${MODEL:-gpt-5.6-luna}"
  --replays "${REPLAYS:-10}" --window-steps "${WINDOW_STEPS:-3}"
  --window-rollouts "${WINDOW_ROLLOUTS:-3}" --replay-concurrency "${REPLAY_CONCURRENCY:-5}" --state "$STATE")
# shellcheck disable=SC2206  # space-separated list
[ -n "${TARGETS:-}" ] && ARGS+=(--targets $TARGETS)
[ -n "${CONDITIONS:-}" ] && ARGS+=(--conditions $CONDITIONS)
[ -n "${DONORS:-}" ] && ARGS+=(--donors $DONORS)
[ "${D1:-0}" = 1 ] && ARGS+=(--d1)
[ "${COUNT_SETUP_CFG:-0}" = 1 ] && ARGS+=(--count-setup-cfg)
PY=("${UV[@]}" python latent_risk/experiments/recovery_context.py)
case "${1:-}" in
  generate|plan|report) "${PY[@]}" "$1" "${ARGS[@]}" ;;
  run|window) "${PY[@]}" "$1" "${ARGS[@]}" 2>&1 | tee -a "$STATE.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
