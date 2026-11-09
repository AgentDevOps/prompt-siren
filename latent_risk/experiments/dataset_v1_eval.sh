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
# Dataset v2 (rq3/framework_report.md 7.2). With MANIFEST (latent_risk/experiments/sample_dataset.py)
# only the manifest's sampled runs of AGENT and CONDITION are used, each with its cell's pooled
# baseline; the task pairs are the manifest's (also for build-images), and the state is a separate
# copy (truncated/self_evolve_td_v2_<agent>_<condition>). Labels are found by run, so earlier
# replays of a sampled run (e.g. this script's skill_only labels) are reused. The chains go to their
# own folder (rq3/task_deviation_v2_<agent>_<condition>); prepare copies in the chains already
# extracted for the manifest's jobs (from rq3/task_deviation for gpt standard), so extract only
# calls the extractor for the rest. Replays reuse the images on disk (REUSE_IMAGES=1): the images
# carry their condition in the tag, and prepare_datasets.sh checks them before jobs reuse them.
# Replay labels for E1.2 and E2.1 come from prevalence_baselines.sh (same MANIFEST, AGENT,
# CONDITION), which labels v1's flags too.
#   MANIFEST=rq3/dataset_v2/manifest.json AGENT=gpt CONDITION=skill_only \
#     latent_risk/experiments/dataset_v1_eval.sh prepare   # then extract, attribute
#
# Images. The injection is written into the SWE-bench image at build time; skill-only images have
# their own tags (...__skill_only), so they can be built and used while jobs of the standard
# condition run. With REUSE_IMAGES=1 (the default for skill_only) replay jobs use the images on
# disk instead of force-rebuilding the shared ones (the source jobs' use_cache=false), so replays
# can run next to other jobs; run build-images once first. Standard images (no suffix) built before
# the condition tags existed may hold a skill-only build: rebuild them once (REBUILD=1, with no job
# running) before standard replays reuse them; REUSE_IMAGES defaults to 0 for standard.
#
# Environment (defaults in brackets): CONDITION [skill_only]  AGENT [gpt] (gpt, qwen, deepseek, muse, glm)
#   MANIFEST [none]  STATE [truncated/self_evolve_td_$CONDITION; with MANIFEST or another AGENT
#   truncated/self_evolve_td_v2_${AGENT}_$CONDITION]  BLIND_DIR [rq3/task_deviation_$CONDITION; for
#   another AGENT rq3/task_deviation_${AGENT}_$CONDITION; with MANIFEST
#   rq3/task_deviation_v2_${AGENT}_$CONDITION]
#   FROM [truncated/self_evolve_td] (frozen v1)  MODEL [gpt-5.6-luna] (extractor)  CONCURRENCY [4]
#   MIN_REPLAYS [5]  MAX_REPLAYS [10]  ZERO_SAFE_AT [10]  REPLAY_CONCURRENCY [3]
#   LABEL_CONCURRENCY [1]  REUSE_IMAGES [1 for skill_only, 0 for standard; 1 with MANIFEST]  REBUILD [0]
#   ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."

CONDITION="${CONDITION:-skill_only}"
AGENT="${AGENT:-gpt}"
MANIFEST="${MANIFEST:-}"
if [ -z "$MANIFEST" ] && [ "$AGENT" = gpt ]; then
  STATE="${STATE:-truncated/self_evolve_td_$CONDITION}"
else
  STATE="${STATE:-truncated/self_evolve_td_v2_${AGENT}_$CONDITION}"
fi
# chains extracted before dataset v2 (the folders without a manifest)
if [ "$AGENT" = gpt ] && [ "$CONDITION" = standard ]; then
  OLD_BLIND_DIR=rq3/task_deviation  # the self-evolving run's chains
elif [ "$AGENT" = gpt ]; then
  OLD_BLIND_DIR="rq3/task_deviation_$CONDITION"
else
  OLD_BLIND_DIR="rq3/task_deviation_${AGENT}_$CONDITION"
fi
if [ -n "$MANIFEST" ]; then
  BLIND_DIR="${BLIND_DIR:-rq3/task_deviation_v2_${AGENT}_$CONDITION}"
elif [ "$AGENT" = gpt ]; then
  BLIND_DIR="${BLIND_DIR:-rq3/task_deviation_$CONDITION}"
else
  BLIND_DIR="${BLIND_DIR:-rq3/task_deviation_${AGENT}_$CONDITION}"
fi
FROM="${FROM:-truncated/self_evolve_td}"
ENV_FILE="${ENV_FILE:-.env}"
if command -v uv >/dev/null; then
  UV=(uv run)
  [ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
else  # no uv on PATH: the project venv, with the env file loaded
  if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE" >/dev/null 2>&1; set +a; fi
  UV=(env PATH="$PWD/.venv/bin:$PATH")
fi
JOB_PREFIX="$("${UV[@]}" python -c "import sys; sys.path.insert(0, 'latent_risk')
from scripts.attribution_replay_study import MODEL_PREFIXES; print(MODEL_PREFIXES['$AGENT'])")"
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
MANIFEST_JOBS=()
if [ -n "$MANIFEST" ]; then  # the manifest's task pairs (as task dirs) and jobs of AGENT, CONDITION
  mapfile -t TASKS < <(python3 -c "import json, sys
cells = [c for c in json.load(open(sys.argv[1]))['cells'] if c['model'] == sys.argv[2] and c['condition'] == sys.argv[3]]
print('\n'.join(sorted({c['task_id'].replace('__', '_', 1).replace(':', '_') for c in cells})))" \
    "$MANIFEST" "$AGENT" "$CONDITION")
  mapfile -t MANIFEST_JOBS < <(python3 -c "import json, sys
for c in json.load(open(sys.argv[1]))['cells']:
    if c['model'] == sys.argv[2] and c['condition'] == sys.argv[3]: print('\n'.join(c['jobs']))" \
    "$MANIFEST" "$AGENT" "$CONDITION")
  [ ${#TASKS[@]} -gt 0 ] || { echo "no $AGENT $CONDITION cells in $MANIFEST" >&2; exit 1; }
fi
case "$CONDITION" in
  skill_only) SUB=/skill_only; REUSE_IMAGES="${REUSE_IMAGES:-1}" ;;
  standard) SUB=""; [ -n "$MANIFEST" ] && REUSE_IMAGES="${REUSE_IMAGES:-1}"
    REUSE_IMAGES="${REUSE_IMAGES:-0}" ;;
  *) echo "CONDITION must be skill_only or standard" >&2; exit 2 ;;
esac

# Every task pair is held out, and no validation split is drawn (nothing is tuned here).
HOLDOUT=("${UV[@]}" python latent_risk/scripts/self_evolve_attribution.py holdout
  --state "$STATE" --workflow task-deviation --blind-dir "$BLIND_DIR"
  --condition "$CONDITION" --pool-baseline
  --exclude-after-attack --eval-all-runs --holdout-tasks "${TASKS[@]}" --versions v1
  --context-after 1 --concurrency "${CONCURRENCY:-4}"
  --min-replays "${MIN_REPLAYS:-5}" --max-replays "${MAX_REPLAYS:-10}"
  --zero-safe-at "${ZERO_SAFE_AT:-10}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}"
  --agent-model "$AGENT")
[ "$REUSE_IMAGES" = 1 ] && HOLDOUT+=(--reuse-images)
# manifest paths are relative to the repository, under jobs/
[ -n "$MANIFEST" ] && HOLDOUT+=(--manifest "$MANIFEST" --jobs-root jobs)

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
    if [ -n "$MANIFEST" ] && [ -d "$OLD_BLIND_DIR" ]; then  # reuse chains (copies)
      copied=0
      for job in "${MANIFEST_JOBS[@]}"; do
        name="$(basename "$job")"
        if [ -d "$OLD_BLIND_DIR/$name" ] && [ ! -e "$BLIND_DIR/$name" ]; then
          mkdir -p "$BLIND_DIR"
          cp -a "$OLD_BLIND_DIR/$name" "$BLIND_DIR/"
          copied=$((copied + 1))
        fi
      done
      echo "chains: $BLIND_DIR ($copied jobs' chains copied from $OLD_BLIND_DIR)"
    fi
    ;;
  fix-split) hold_out_all ;;  # for a state prepared before hold_out_all existed
  extract)
    jobs=("${MANIFEST_JOBS[@]}")
    if [ -z "$MANIFEST" ]; then
      for task in "${TASKS[@]}"; do
        for job in jobs/groups/"$task$SUB"/"$JOB_PREFIX"*; do
          [ -d "$job" ] && jobs+=("$job")
        done
      done
    fi
    runs=()
    if [ -n "$MANIFEST" ]; then  # only the sampled attack-failed runs (the population)
      mapfile -t runs < <(python3 -c "import json, os, sys
for c in json.load(open(sys.argv[1]))['cells']:
    if c['model'] == sys.argv[2] and c['condition'] == sys.argv[3]:
        print('\n'.join(os.path.basename(r['path']) for r in c['runs'] if not r['attack_success']))" \
        "$MANIFEST" "$AGENT" "$CONDITION")
      runs=(--run-ids "${runs[@]}")
    fi
    echo "${#jobs[@]} jobs of ${#TASKS[@]} task pairs ($AGENT, $CONDITION)"
    "${UV[@]}" python latent_risk/scripts/task_deviation_attribution.py "${jobs[@]}" "${runs[@]}" \
      --output-dir "$BLIND_DIR" --model "${MODEL:-gpt-5.6-luna}" --extract-only \
      --concurrency "${CONCURRENCY:-4}" 2>&1 | tee -a "$BLIND_DIR.log"
    ;;
  attribute) "${HOLDOUT[@]}" 2>&1 | tee -a "$STATE.log" ;;
  build-images)
    rebuild=()
    [ "${REBUILD:-0}" = 1 ] && rebuild=(--rebuild)
    pairs=()
    if [ -n "$MANIFEST" ]; then
      mapfile -t pairs < <(python3 -c "import json, sys
print('\n'.join(sorted({c['task_id'] for c in json.load(open(sys.argv[1]))['cells'] if c['condition'] == sys.argv[2]})))" \
        "$MANIFEST" "$CONDITION")
      pairs=(--pairs "${pairs[@]}")
    fi
    "${UV[@]}" python latent_risk/experiments/build_condition_images.py \
      --condition "$CONDITION" "${rebuild[@]}" "${pairs[@]}"
    ;;
  label)
    part=()
    [ -n "${2:-}" ] && part=(--part "$2")
    "${HOLDOUT[@]}" --label --label-concurrency "${LABEL_CONCURRENCY:-1}" "${part[@]}" \
      2>&1 | tee -a "$STATE.log"
    ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
