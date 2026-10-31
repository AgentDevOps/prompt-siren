#!/usr/bin/env bash
# The frozen payload-blind workflow on a new gpt-5.6-luna dataset (step P0 of the paper plan,
# rq3/framework_report.md section 7.3, then the replay labels of v1's flags).
#
# CONDITION selects the jobs: skill_only = the jobs in jobs/groups/<pair>/skill_only/. Nothing is
# tuned on them: the task-deviation extractor and judge v1 at its threshold frozen on the training
# split of truncated/self_evolve_td are applied unchanged, so every run is held out. The jobs of a
# task pair are pooled into one fresh-run baseline (--pool-baseline).
#
#   latent_risk/experiments/dataset_v1_eval.sh prepare      # state folder: frozen v1 (copy, no replays)
#   latent_risk/experiments/dataset_v1_eval.sh extract      # task-deviation chains (LLM calls only)
#   latent_risk/experiments/dataset_v1_eval.sh attribute    # v1 scores and flags, replay cost (LLM calls only)
#   latent_risk/experiments/dataset_v1_eval.sh build-images # this condition's images (Docker, no LLM)
#   latent_risk/experiments/dataset_v1_eval.sh label [K/N]  # replay-label v1's flags (optional share)
#
# Results: ${STATE}_holdout/report.md (v1's flags and labels), chains in $BLIND_DIR.
#
# Images. The injection is written into the SWE-bench image at build time; skill-only images have
# their own tags (...__skill_only), so they can be built and used while jobs of the standard
# condition run. With REUSE_IMAGES=1 (the default for skill_only) replay jobs use the images on
# disk instead of force-rebuilding the shared ones (the source jobs' use_cache=false), so replays
# can run next to other jobs; run build-images once first. Standard images (no suffix) built before
# the condition tags existed may hold a skill-only build: rebuild them once (REBUILD=1, with no job
# running) before standard replays reuse them; REUSE_IMAGES defaults to 0 for standard.
#
# Environment (defaults in brackets): CONDITION [skill_only]
#   STATE [truncated/self_evolve_td_$CONDITION]  BLIND_DIR [rq3/task_deviation_$CONDITION]
#   FROM [truncated/self_evolve_td] (frozen v1)  MODEL [gpt-5.6-luna] (extractor)  CONCURRENCY [4]
#   MIN_REPLAYS [5]  MAX_REPLAYS [10]  ZERO_SAFE_AT [10]  REPLAY_CONCURRENCY [3]
#   LABEL_CONCURRENCY [1]  REUSE_IMAGES [1 for skill_only, 0 for standard]  REBUILD [0]
#   ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."

CONDITION="${CONDITION:-skill_only}"
STATE="${STATE:-truncated/self_evolve_td_$CONDITION}"
BLIND_DIR="${BLIND_DIR:-rq3/task_deviation_$CONDITION}"
FROM="${FROM:-truncated/self_evolve_td}"
ENV_FILE="${ENV_FILE:-.env}"
if command -v uv >/dev/null; then
  UV=(uv run)
  [ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
else  # no uv on PATH: the project venv, with the env file loaded
  if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE" >/dev/null 2>&1; set +a; fi
  UV=(env PATH="$PWD/.venv/bin:$PATH")
fi
JOB_PREFIX="swebench_mini_swe_openai_gpt-5_6-luna_template_string_"
TASKS=(
  astropy_astropy-14182_disable_security_tests_task
  astropy_astropy-14182_skip_error_handling_task
  astropy_astropy-14995_dependency_poisoning_task
  sympy_sympy-15345_modify_readme_curl_bash_task
  sympy_sympy-15345_skip_error_handling_task
  sympy_sympy-17139_disable_security_tests_task
  sympy_sympy-17139_modify_readme_curl_bash_task
  sympy_sympy-17139_skip_error_handling_task
)
case "$CONDITION" in
  skill_only) SUB=/skill_only; REUSE_IMAGES="${REUSE_IMAGES:-1}" ;;
  standard) SUB=""; REUSE_IMAGES="${REUSE_IMAGES:-0}" ;;
  *) echo "CONDITION must be skill_only or standard" >&2; exit 2 ;;
esac

# Every task pair is held out, and no validation split is drawn (nothing is tuned here).
HOLDOUT=("${UV[@]}" python latent_risk/scripts/self_evolve_attribution.py holdout
  --state "$STATE" --workflow task-deviation --blind-dir "$BLIND_DIR"
  --condition "$CONDITION" --pool-baseline
  --exclude-after-attack --eval-all-runs --holdout-tasks "${TASKS[@]}" --versions v1
  --context-after 1 --concurrency "${CONCURRENCY:-4}"
  --min-replays "${MIN_REPLAYS:-5}" --max-replays "${MAX_REPLAYS:-10}"
  --zero-safe-at "${ZERO_SAFE_AT:-10}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}")
[ "$REUSE_IMAGES" = 1 ] && HOLDOUT+=(--reuse-images)

hold_out_all() {  # the copy's recorded split: every task pair of this dataset is held out
  "${UV[@]}" python - "$STATE/settings.json" "${TASKS[@]}" <<'PY'
import json, sys
path, tasks = sys.argv[1], sorted(sys.argv[2:])
settings = json.load(open(path))
if settings.get("holdout_tasks") != tasks:
    settings["holdout_tasks"] = tasks
    json.dump(settings, open(path, "w"), indent=2)
    print(f"{path}: holdout_tasks = all {len(tasks)} task pairs")
PY
}

case "${1:-}" in
  prepare)
    if [ -e "$STATE" ]; then
      echo "$STATE exists already" >&2
      exit 1
    fi
    # replays excluded: labels are read from all of truncated/, a copy would count them twice
    rsync -a --exclude /replays "$FROM/" "$STATE/"
    hold_out_all
    echo "state: $STATE (v1 threshold and judge settings from $FROM)"
    ;;
  fix-split) hold_out_all ;;  # for a state prepared before hold_out_all existed
  extract)
    jobs=()
    for task in "${TASKS[@]}"; do
      for job in jobs/groups/"$task$SUB"/"$JOB_PREFIX"*; do
        [ -d "$job" ] && jobs+=("$job")
      done
    done
    echo "${#jobs[@]} jobs of ${#TASKS[@]} task pairs ($CONDITION)"
    "${UV[@]}" python latent_risk/scripts/task_deviation_attribution.py "${jobs[@]}" \
      --output-dir "$BLIND_DIR" --model "${MODEL:-gpt-5.6-luna}" --extract-only \
      --concurrency "${CONCURRENCY:-4}" 2>&1 | tee -a "$BLIND_DIR.log"
    ;;
  attribute) "${HOLDOUT[@]}" 2>&1 | tee -a "$STATE.log" ;;
  build-images)
    rebuild=()
    [ "${REBUILD:-0}" = 1 ] && rebuild=(--rebuild)
    "${UV[@]}" python latent_risk/experiments/build_condition_images.py \
      --condition "$CONDITION" "${rebuild[@]}"
    ;;
  label)
    part=()
    [ -n "${2:-}" ] && part=(--part "$2")
    "${HOLDOUT[@]}" --label --label-concurrency "${LABEL_CONCURRENCY:-1}" "${part[@]}" \
      2>&1 | tee -a "$STATE.log"
    ;;
  *) sed -n '2,31p' "$0"; exit 1 ;;
esac
