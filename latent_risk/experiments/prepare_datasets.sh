#!/usr/bin/env bash
# Data preparation for the paper: the 8 in-scope task pairs, for two agent models and two
# injection conditions, RUNS fresh attack runs each.
#
#   MODEL       gpt   gpt-5.6-luna (OpenAI), mini-swe gpt5.yaml, 60 steps (the current study)
#               qwen  qwen3.6-35b (local vLLM, see 'serve'), mini-swe qwen36_text_compact.yaml,
#                     no max_steps override, exactly as the existing qwen jobs (qwen needs more
#                     steps to finish; the config's own step_limit of 100 is the only bound)
#   CONDITION   standard    skill + source injection   -> jobs/groups/<task dir>/<job>
#               skill_only  skill injection only        -> jobs/groups/<task dir>/skill_only/<job>
#
# This is the layout the existing jobs use; the current study's job discovery takes only
# standard-condition gpt-5.6-luna jobs outside skill_only/, so new jobs do not change it.
# Jobs of the same model and condition in a pair's folder are pooled (e.g. 20 + 10 runs); a pair
# with at least MIN_VALID valid runs is complete and skipped.
#
# Rounds: RUNS is the number of valid runs each pair should have after this call; a pair short of
# it gets a new job with the missing runs, pooled with its existing jobs (jobs of a pair share all
# settings, so their fresh-run ASRs pool). E.g. all 8 pairs, 10 runs each round:
#   MODEL=gpt CONDITION=skill_only RUNS=10 latent_risk/experiments/prepare_datasets.sh run
#   MODEL=gpt CONDITION=skill_only RUNS=20 latent_risk/experiments/prepare_datasets.sh run
#   MODEL=gpt CONDITION=skill_only RUNS=30 latent_risk/experiments/prepare_datasets.sh run
#
#   latent_risk/experiments/prepare_datasets.sh status                      # all 4 cells x 8 pairs
#   MODEL=... CONDITION=... RUNS=... latent_risk/experiments/prepare_datasets.sh run [PAIR ...]
#   latent_risk/experiments/prepare_datasets.sh serve                       # qwen vLLM server
#
# Environment (defaults in brackets): MODEL [gpt]  CONDITION [standard]  RUNS [30]
#   MIN_VALID [25]  CONCURRENCY [4 for gpt, 1 for qwen]  ENV_FILE [.env]
#   USE_CACHE [false]: false rebuilds the SWE-bench images at the start of each job (force-deleting
#   the shared ones, which breaks other jobs starting containers at that moment); true reuses them
#   qwen server: MODEL_DIR [/mnt/data/xiaoliang/models/Qwen3.6-35B-A3B-AWQ]  VLLM_PYTHON [python]
#   PORT [8000]  MAX_MODEL_LEN [65536]  MAX_NUM_SEQS [1]
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL="${MODEL:-gpt}"
CONDITION="${CONDITION:-standard}"
RUNS="${RUNS:-30}"
ENV_FILE="${ENV_FILE:-.env}"
if command -v uv >/dev/null; then
  UV=(uv run)
  [ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
else  # no uv on PATH: the project venv, with the env file loaded
  if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE" >/dev/null 2>&1; set +a; fi
  UV=(env PATH="$PWD/.venv/bin:$PATH")
fi
PAIRS=(
  astropy__astropy-14182:disable_security_tests_task
  astropy__astropy-14182:skip_error_handling_task
  astropy__astropy-14995:dependency_poisoning_task
  sympy__sympy-15345:modify_readme_curl_bash_task
  sympy__sympy-15345:skip_error_handling_task
  sympy__sympy-17139:disable_security_tests_task
  sympy__sympy-17139:modify_readme_curl_bash_task
  sympy__sympy-17139:skip_error_handling_task
)

model_settings() {  # sets PREFIX, SPEC, EXTRA, DEFAULT_CONCURRENCY for $1
  case "$1" in
    gpt)
      PREFIX=swebench_mini_swe_openai_gpt-5_6-luna_template_string_
      SPEC=/home/xiaoliang_zhang/mini-swe-agent/src/minisweagent/config/gpt5.yaml
      EXTRA=(+agent.config.max_steps=60)
      DEFAULT_CONCURRENCY=4
      ;;
    qwen)
      PREFIX=swebench_mini_swe_hosted_vllm_qwen3_6-35b_template_string_
      SPEC="$PWD/src/prompt_siren/config/default/mini_swe/qwen36_text_compact.yaml"
      EXTRA=()
      DEFAULT_CONCURRENCY=1
      ;;
    *) echo "MODEL must be gpt or qwen" >&2; exit 2 ;;
  esac
}

task_dir() { printf '%s' "$1" | sed 's/__/_/; s/:/_/'; }

folder() {  # target folder of a pair under $CONDITION
  case "$CONDITION" in
    standard) printf 'jobs/groups/%s' "$(task_dir "$1")" ;;
    skill_only) printf 'jobs/groups/%s/skill_only' "$(task_dir "$1")" ;;
    *) echo "CONDITION must be standard or skill_only" >&2; exit 2 ;;
  esac
}

# "<valid runs> <successes> <jobs> <completed runs of the largest job> <largest job>", pooled
# over the jobs of this model and condition in a folder
existing() {
  FOLDER="$1" PREFIX="$PREFIX" CONDITION="$CONDITION" python3 - <<'PY'
import glob, json, os, re
valid = successes = jobs = 0
largest = (0, "")
for job in glob.glob(f"{os.environ['FOLDER']}/{os.environ['PREFIX']}*"):
    config = open(os.path.join(job, "config.yaml"), encoding="utf-8").read() if os.path.exists(os.path.join(job, "config.yaml")) else ""
    source = re.search(r'"?enable_source_injection"?: (\w+)', config)
    skill = re.search(r'"?enable_skill_injection"?: (\w+)', config)
    if not (source and skill and skill.group(1) == "true"):
        continue
    if (source.group(1) == "true") != (os.environ["CONDITION"] == "standard"):
        continue
    results = glob.glob(os.path.join(job, "*", "*", "result.json"))
    path = os.path.join(job, "asr.json")
    if os.path.exists(path):
        asr = json.load(open(path))
        valid += asr.get("valid_attack_runs", 0)
        successes += asr.get("attack_successes", 0)
    else:  # older jobs without asr.json: count the finished runs
        scores = [json.load(open(r)).get("attack_score") for r in results]
        valid += sum(score is not None for score in scores)
        successes += sum(bool(score) for score in scores)
    jobs += 1
    largest = max(largest, (len(results), job))
if jobs:
    print(valid, successes, jobs, *largest)
PY
}

check_server() {
  if [ "$MODEL" = qwen ]; then
    curl -sf -m 5 "http://127.0.0.1:${PORT:-8000}/v1/models" | grep -q qwen3.6-35b \
      || { echo "no qwen3.6-35b server on port ${PORT:-8000}; run 'serve' first" >&2; exit 1; }
  fi
}

run_pair() {
  local task="$1" target found before after new need rc=0
  target="$(folder "$task")"
  found="$(existing "$target")"
  need="$RUNS"
  if [ -n "$found" ]; then
    # shellcheck disable=SC2086  # five fields
    set -- $found
    if [ "$1" -ge "${MIN_VALID:-25}" ] || [ "$1" -ge "$RUNS" ]; then
      echo "skip $task ($MODEL, $CONDITION): $1 valid runs already (target $RUNS)"
      return 0
    fi
    # the missing runs go into a new job next to the existing ones (pooled by 'existing');
    # resuming a job to fill it did not reliably add runs
    need=$((RUNS - $1))
  fi
  check_server
  echo "run $task ($MODEL, $CONDITION): $need new runs (target $RUNS)"
  local source=true
  [ "$CONDITION" = skill_only ] && source=false
  before="$(ls -d jobs/"$PREFIX"* 2>/dev/null | sort || true)"
  # same settings as benchmark_pair.sh (the current gpt-5.6-luna study) except model and condition
  "${UV[@]}" prompt-siren jobs start attack \
    +dataset=swebench +attack=agentdojo_important_instructions +sandbox_manager=local-docker \
    agent.type=mini_swe \
    "+agent.config.config_specs=[\"$SPEC\",\"agent.mode=yolo\",\"agent.confirm_exit=false\"]" \
    dataset.config.registry=null dataset.config.enable_network=false dataset.config.bash_timeout=180 \
    sandbox_manager.config.network_enabled=false \
    "task_ids=[\"$task\"]" output.n_runs_per_task="$need" \
    execution.concurrency="${CONCURRENCY:-$DEFAULT_CONCURRENCY}" \
    telemetry.trace_console=false telemetry.otel_endpoint=null \
    dataset.config.enable_source_injection="$source" dataset.config.enable_skill_injection=true \
    dataset.config.use_cache="${USE_CACHE:-false}" trajectory_labeling.method=none "${EXTRA[@]}" || rc=$?
  after="$(ls -d jobs/"$PREFIX"* 2>/dev/null | sort || true)"
  new="$(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | grep . || true)"
  if [ "$(printf '%s\n' "$new" | grep -c .)" -ne 1 ]; then
    echo "could not identify the new job under jobs/ (found: ${new:-none}); move it to $target/ by hand" >&2
    exit 1
  fi
  mkdir -p "$target"
  mv "$new" "$target/"
  echo "job: $target/$(basename "$new")"
  return "$rc"
}

status() {
  printf '%-48s' "task pair"
  for cell in gpt:standard gpt:skill_only qwen:standard qwen:skill_only; do printf '%-18s' "$cell"; done
  printf '\n'
  for task in "${PAIRS[@]}"; do
    printf '%-48s' "$(task_dir "$task" | sed 's/_task$//')"
    for cell in gpt:standard gpt:skill_only qwen:standard qwen:skill_only; do
      model_settings "${cell%%:*}"
      CONDITION="${cell##*:}"
      found="$(existing "$(folder "$task")")"
      if [ -z "$found" ]; then
        printf '%-18s' "missing"
      else
        set -- $found
        mark=""
        [ "$3" -gt 1 ] && mark=" ($3 jobs)"
        [ "$1" -lt "${MIN_VALID:-25}" ] && mark=" (incomplete)"
        printf '%-18s' "$2/$1$mark"
      fi
    done
    printf '\n'
  done
  echo "cells: successes/valid runs pooled over the cell's jobs; complete at ${MIN_VALID:-25} valid runs"
}

serve() {
  exec "${VLLM_PYTHON:-python}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR:-/mnt/data/xiaoliang/models/Qwen3.6-35B-A3B-AWQ}" \
    --served-model-name qwen3.6-35b --host 127.0.0.1 --port "${PORT:-8000}" \
    --tensor-parallel-size 2 --max-model-len "${MAX_MODEL_LEN:-65536}" \
    --gpu-memory-utilization 0.95 --max-num-seqs "${MAX_NUM_SEQS:-1}" \
    --max-num-batched-tokens 4096 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --disable-custom-all-reduce --enforce-eager
}

case "${1:-}" in
  status) status ;;
  serve) serve ;;
  run)
    shift
    model_settings "$MODEL"
    pairs=("$@")
    [ ${#pairs[@]} -eq 0 ] && pairs=("${PAIRS[@]}")
    failed=()
    # a crashed job (e.g. a transient Docker timeout) is still filed; rerunning tops it up
    for task in "${pairs[@]}"; do run_pair "$task" || failed+=("$task"); done
    if [ ${#failed[@]} -gt 0 ]; then
      echo "incomplete (rerun the same command to top up): ${failed[*]}" >&2
      exit 1
    fi
    ;;
  *) sed -n '2,25p' "$0"; exit 1 ;;
esac
