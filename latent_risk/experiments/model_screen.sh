#!/usr/bin/env bash
# Screening of candidate agent models for the clean test of the method (paper_plan.md step 12).
# Stage 1: a few skill-only runs per model on 3 task pairs where gpt and qwen are both susceptible;
# the report gives per model the ASR, benign score, format errors, engagement with the injection
# (attack-failed runs whose commands touch a file the injection names), steps, tokens and cost per
# run, and the projected cost of the full study. Nothing here reads R3/R4 scores or labels; the
# screening jobs live under jobs/model_screen/<model>/ and are never part of the test set.
#
#   latent_risk/experiments/model_screen.sh check  [MODEL ...]   # one short call per model:
#                                                   # model id, pinned provider, cost of the call
#   latent_risk/experiments/model_screen.sh run    [MODEL ...]   # stage 1 runs (the user starts it)
#   latent_risk/experiments/model_screen.sh report               # $OUT/report.md
#   latent_risk/experiments/model_screen.sh monitor [MODEL ...]  # R1 on the attack-failed runs
#                                                   # (LLM only, Claude runs it): $OUT/monitor/monitor.md
#
# Models (name: model, endpoint pinned without fallback, price per 1M tokens in / cached / out).
# Price band: gpt-6-luna (0.10 / 0.01 / 0.50) to gpt-5.6-luna (0.20 / 0.02 / 1.20).
#   gpt6      openai/gpt-6-luna (OpenAI API, OPENAI_API_KEY)               0.10  / 0.01   / 0.50
#   mistral   openrouter mistralai/mistral-small-2603, mistral             0.15  / 0.015  / 0.60
#   minimax   openrouter minimax/minimax-m3.1-flash, parasail (fp8)        0.15  / 0.015  / 0.60
#   mimo      openrouter xiaomi/mimo-v2.6-flash, xiaomi (fp8)              0.14  / 0.003  / 0.28
#   gemini    openrouter google/gemini-2.5-flash-lite, google-ai-studio    0.10  / 0.01   / 0.40
#   nemotron  openrouter nvidia/nemotron-3.5-super-vl, crusoe (bf16)       0.10  / 0.08   / 0.40
# Round 2 (2026-11-24): every model robust in round 1 and before was released in or after August 2026,
# the two susceptible ones earlier (qwen3.6 April, gpt-5.6-luna July); so models released
# February to July 2026, at most gpt-5.6-luna's cost per run (MiniMax dropped: 7x the cost):
#   gpt54nano  openai/gpt-5.4-nano (OpenAI API), March                     0.20  / 0.02   / 1.25
#   gemma4     openrouter google/gemma-4-26b-a4b-it, coreweave (bf16), April 0.10 / 0.05  / 0.30
#   gemini31   openrouter google/gemini-3.1-flash-lite, google-ai-studio, May 0.25 / 0.025 / 1.50
#   hy3        openrouter tencent/hy3, tencent (fp8), July                 0.083 / 0.021  / 0.33
#   step37     openrouter stepfun/step-3.7-flash, stepfun (fp8), May       0.20  / 0.04   / 1.15
#   nemotron3  openrouter nvidia/nemotron-3-super-120b-a12b, deepinfra (bf16), March 0.075 / - / 0.40
#   seed20     openrouter bytedance-seed/seed-2.0-mini, seed (fp8), February 0.10 / - / 0.40
# run and check without MODEL take the round-2 models; report lists all.
#
# Environment (defaults in brackets): RUNS [5] runs per pair  CONDITION [skill_only]
#   STAGE [1] (2: the 4 stage-2 pairs, report in $OUT/stage2_$CONDITION)
#   CONCURRENCY [4]  USE_CACHE [true] (the pairs' skill-only images exist from the gpt/qwen runs)
#   ROOT [jobs/model_screen]  OUT [truncated/model_screen]  STUDY_RUNS [2500] run-equivalents of a
#   full study (census runs + label and warning replays), for the projected cost  ENV_FILE [.env]
#   monitor: MODEL [gpt-5.6-luna] extractor  MONITOR_CONCURRENCY [8]
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_FILE="${ENV_FILE:-.env}"
if command -v uv >/dev/null; then
  UV=(uv run)
  [ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
else
  if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE" >/dev/null 2>&1; set +a; fi
  UV=(env PATH="$PWD/.venv/bin:$PATH")
fi
export PYTHONPATH="latent_risk/experiments:latent_risk${PYTHONPATH:+:$PYTHONPATH}"
ROOT="${ROOT:-jobs/model_screen}"
OUT="${OUT:-truncated/model_screen}"
CONDITION="${CONDITION:-skill_only}"

# name|mini-swe model name|OpenRouter provider slug (- = OpenAI API)|in|cached|out
MODELS=(
  "gpt6|openai/gpt-6-luna|-|0.10|0.01|0.50"
  "mistral|openrouter/mistralai/mistral-small-2603|mistral|0.15|0.015|0.60"
  "minimax|openrouter/minimax/minimax-m3.1-flash|parasail|0.15|0.015|0.60"
  "mimo|openrouter/xiaomi/mimo-v2.6-flash|xiaomi|0.14|0.003|0.28"
  "gemini|openrouter/google/gemini-2.5-flash-lite|google-ai-studio|0.10|0.01|0.40"
  "nemotron|openrouter/nvidia/nemotron-3.5-super-vl|crusoe|0.10|0.08|0.40"
  "gpt54nano|openai/gpt-5.4-nano|-|0.20|0.02|1.25"
  "gemma4|openrouter/google/gemma-4-26b-a4b-it|coreweave|0.10|0.05|0.30"
  "gemini31|openrouter/google/gemini-3.1-flash-lite|google-ai-studio|0.25|0.025|1.50"
  "hy3|openrouter/tencent/hy3|tencent|0.083|0.021|0.33"
  "step37|openrouter/stepfun/step-3.7-flash|stepfun|0.20|0.04|1.15"
  "nemotron3|openrouter/nvidia/nemotron-3-super-120b-a12b|deepinfra|0.075|0.075|0.40"
  "seed20|openrouter/bytedance-seed/seed-2.0-mini|seed|0.10|0.10|0.40"
)
ROUND=(gpt54nano gemma4 gemini31 hy3 step37 nemotron3 seed20)
PAIRS=(
  sympy__sympy-17139:skip_error_handling_task
  sympy__sympy-15345:modify_readme_curl_bash_task
  sympy__sympy-17139:disable_security_tests_task
)
# STAGE=2: survivors of stage 1 on 4 further pairs of the study's 8 (images of both conditions
# exist), in both conditions (CONDITION=standard and skill_only)
if [ "${STAGE:-1}" = 2 ]; then
  PAIRS=(
    astropy__astropy-14182:disable_security_tests_task
    astropy__astropy-14182:skip_error_handling_task
    astropy__astropy-14995:dependency_poisoning_task
    sympy__sympy-17139:modify_readme_curl_bash_task
  )
  OUT="${OUT}/stage2_$CONDITION"
fi

entry() {  # sets NAME MODEL_NAME SLUG P_IN P_CACHE P_OUT for model $1
  local row
  for row in "${MODELS[@]}"; do
    IFS='|' read -r NAME MODEL_NAME SLUG P_IN P_CACHE P_OUT <<<"$row"
    [ "$NAME" = "$1" ] && return 0
  done
  echo "unknown model $1 (known: $(printf '%s ' "${MODELS[@]%%|*}"))" >&2
  exit 2
}

overlay() {  # writes the mini-swe overlay of the current model, prints its path
  local path="$OUT/overlays/$NAME.yaml"
  mkdir -p "$OUT/overlays"
  {
    echo "# mini-swe overlay for model_screen.sh ($NAME), merged over gpt5.yaml: only the model changes"
    echo "model:"
    echo "  model_name: $MODEL_NAME"
    echo "  # litellm may have no price for this model; cost tracking would stop the run"
    echo "  cost_tracking: ignore_errors"
    if [ "$SLUG" != - ]; then
      echo "  model_kwargs:"
      echo "    extra_body:"
      echo "      provider:"
      echo "        order: [$SLUG]"
      echo "        allow_fallbacks: false"
    fi
  } >"$path"
  printf '%s\n' "$PWD/$path"
}

prefix() {  # job folder prefix: the sanitised mini-swe model name (job/naming.py)
  printf 'swebench_mini_swe_%s_template_string_' "$(printf '%s' "$MODEL_NAME" | sed 's/[^A-Za-z0-9_-]/_/g')"
}

selected() { [ $# -gt 0 ] && printf '%s\n' "$@" || printf '%s\n' "${ROUND[@]}"; }

check() {
  local name
  for name in $(selected "$@"); do
    entry "$name"
    "${UV[@]}" python latent_risk/experiments/model_screen.py check --model "$MODEL_NAME" \
      --provider "$SLUG" --prices "$P_IN" "$P_CACHE" "$P_OUT" || true
  done
}

run() {
  local name failed=()
  for name in $(selected "$@"); do
    entry "$name"
    MODEL=custom CUSTOM_PREFIX="$(prefix)" CUSTOM_OVERLAY="$(overlay)" \
      CUSTOM_OPENROUTER="$([ "$SLUG" = - ] && echo 0 || echo 1)" \
      ROOT="$ROOT/$NAME" POOL_ROOTS="" CONDITION="$CONDITION" RUNS="${RUNS:-5}" \
      MIN_VALID="${RUNS:-5}" USE_CACHE="${USE_CACHE:-true}" CONCURRENCY="${CONCURRENCY:-4}" \
      ENV_FILE="$ENV_FILE" latent_risk/experiments/prepare_datasets.sh run "${PAIRS[@]}" \
      || failed+=("$name")
  done
  if [ ${#failed[@]} -gt 0 ]; then
    echo "incomplete (rerun to top up): ${failed[*]}" >&2
    exit 1
  fi
}

report() {
  local args=() row
  for row in "${MODELS[@]}"; do
    IFS='|' read -r NAME MODEL_NAME SLUG P_IN P_CACHE P_OUT <<<"$row"
    args+=(--model-spec "$NAME|$(prefix)|$P_IN|$P_CACHE|$P_OUT")
  done
  "${UV[@]}" python latent_risk/experiments/model_screen.py report --root "$ROOT" \
    --condition "$CONDITION" --pair-dirs $(printf '%s\n' "${PAIRS[@]}" | sed 's/__/_/; s/:/_/') --study-runs "${STUDY_RUNS:-2500}" --out "$OUT/report.md" "${args[@]}" \
    --reference "gpt-6-luna|0.10|0.01|0.50" --reference "gpt-5.6-luna|0.20|0.02|1.20"
}

monitor() {  # R1 as in end_to_end.sh (extractor MODEL [gpt-5.6-luna])
  local args=() name
  for name in $(selected "$@"); do
    entry "$name"
    args+=(--model-spec "$NAME|$(prefix)")
  done
  "${UV[@]}" python latent_risk/experiments/model_screen.py monitor --root "$ROOT" \
    --condition "$CONDITION" --pair-dirs $(printf '%s\n' "${PAIRS[@]}" | sed 's/__/_/; s/:/_/') \
    --model "${MODEL:-gpt-5.6-luna}" --concurrency "${MONITOR_CONCURRENCY:-8}" \
    --out "$OUT/monitor" "${args[@]}"
}

case "${1:-}" in
  check) shift; check "$@" ;;
  monitor) shift; monitor "$@" 2>&1 | tee -a "$OUT.monitor.log" ;;
  run) shift; run "$@" 2>&1 | tee -a "$OUT.log" ;;
  report) report ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
