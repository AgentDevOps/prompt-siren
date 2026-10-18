#!/usr/bin/env bash
# Self-evolving attribution, round 8: round 7 with behaviour-based protection.
#
# The only change from round 7: rules are no longer frozen word for word when they support a correct
# pick (--protect behaviour). Any rule may be revised or removed; instead, the verified dangerous
# messages the incumbent and v1 catch form a regression set that may lose at most one message, and
# the optimiser sees each rule's provenance (the cases it was added for, and whether that evidence
# still holds). Round 7's test and validation splits are copied, so results compare directly.
#
# From round 7 (dataset: the 8 task pairs that have verified dangerous messages):
# All three sets are drawn at random (seeded) and stored in the state folder:
#   test        TEST_TASKS task pairs with >= TEST_MIN_POSITIVES verified dangerous messages
#               (test_split.json; used only by holdout/curve)
#   validation  VALIDATION_TASKS task pairs from the rest (validation_split.json; gate only)
#   training    everything else
#
# From round 6:
# Round 5 showed the remaining problem is overfitting: v4a raised training precision but lost the
# held-out positives. Round 6 adds a validation set: task pairs drawn at random (seeded) from the
# non-held-out ones, never trained on. A candidate that passes the training check is rejected if,
# on validation, it catches fewer verified dangerous messages or has a lower precision than the
# incumbent; the outcome (with excerpts of lost positives / new false alarms) is fed back to the
# optimiser. The held-out test task pairs stay untouched for the final result.
#
# Unchanged from round 5:
#   fix 1  the loop is scored on *every* failed training run, not only on runs that already had
#          a label; before iteration 1 it verifies v1's own flags by replay, afterwards each new
#          pick as usual (labels that already exist are reused, not replayed again)
#   fix 2  attack actions (harm signature) and later messages are not selectable candidates,
#          so they can't push the earlier, replay-relevant messages out of the top 3
#   fix 4  six task pairs are held out *before* evolving and never seen by the loop
#
#   experiments/self_evolve_round8.sh run              # verify v1's flags, then evolve from v1
#   experiments/self_evolve_round8.sh status
#   experiments/self_evolve_round8.sh holdout-cost     # held-out: attribute, print replay cost
#   experiments/self_evolve_round8.sh holdout [K/N]    # held-out: replay-label flags + report
#   experiments/self_evolve_round8.sh curve-cost       # learning curve: every accepted version
#   experiments/self_evolve_round8.sh curve [K/N]      #   on the held-out task pairs (overfitting
#                                                      #   check: does held-out follow training?)
#   experiments/self_evolve_round8.sh export           # write all replay labels into jobs/groups
#
# Results: $STATE/log.md (loop), ${STATE}_holdout/report.md (held-out).
#
# Environment (defaults in brackets):
#   STATE [truncated/self_evolve_r8]  MAX_ITERATIONS [18]  PATIENCE [8]
#   SPLIT_FROM [truncated/self_evolve_r7]  state folder whose test/validation splits are reused
#   TEST_TASKS [2]  TEST_SEED [0]  TEST_MIN_POSITIVES [3]  random test draw
#   VALIDATION_TASKS [2]  VALIDATION_SEED [0]  random validation draw
#   LABEL_BUDGET [10]  new picks replay-labelled per iteration
#   VERSIONS (holdout; default v1 and the final incumbent)
#   LABEL_CONCURRENCY [1]  (1 avoids parallel Docker image builds racing each other)
#   MIN_REPLAYS [5]  MAX_REPLAYS [10]  ZERO_SAFE_AT [10]  replays per labelled message
#     (development setting; use 5/30/10 for verification-grade labels). Every command uses the
#     same rule, so training and held-out labels stay comparable.
#   REPLAY_CONCURRENCY [3]  replays of one message run in parallel (same Docker image)
#   ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/.."

STATE="${STATE:-truncated/self_evolve_r8}"
LABEL_CONCURRENCY="${LABEL_CONCURRENCY:-1}"
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
EVOLVE=("${UV[@]}" --extra topics python scripts/self_evolve_attribution.py)

# Reuse round 7's splits (same test and validation task pairs).
SPLIT_FROM="${SPLIT_FROM:-truncated/self_evolve_r7}"
for split in test_split.json validation_split.json; do
  if [ ! -f "$STATE/$split" ] && [ -f "$SPLIT_FROM/$split" ]; then
    mkdir -p "$STATE"
    cp "$SPLIT_FROM/$split" "$STATE/$split"
  fi
done
# ...and its v1 attributions (identical settings), so both rounds start from the same baseline.
for folder in attributions/v1 validation/v1; do
  if [ ! -d "$STATE/$folder" ] && [ -d "$SPLIT_FROM/$folder" ]; then
    mkdir -p "$STATE/$(dirname "$folder")"
    cp -r "$SPLIT_FROM/$folder" "$STATE/$folder"
  fi
done

# Test task pairs: drawn once (seeded) among task pairs of the current dataset, redrawn until they
# hold TEST_MIN_POSITIVES verified dangerous messages; stored so the split never changes.
if [ ! -f "$STATE/test_split.json" ]; then
  mkdir -p "$STATE"
  "${UV[@]}" python - "$STATE/test_split.json" "${TEST_TASKS:-2}" "${TEST_SEED:-0}" \
    "${TEST_MIN_POSITIVES:-3}" <<'PY'
import json, random, sys
from collections import Counter
sys.path.insert(0, ".")
from scripts import self_evolve_attribution as se
path, count, seed, minimum = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
args = se.parse_args(["status"])
sources = se.Sources(args)
labels = se.collect_labels(sources, args.truncated, args.label_rule, exclude_holdout=False)
positives = Counter(
    sources.job_of[key].task_dir for (key, _), label in labels.items() if label.verdict == "dangerous"
)
pool = sorted({job.task_dir for job in sources.job_of.values()})
rng = random.Random(seed)
for attempt in range(1, 1001):
    tasks = sorted(rng.sample(pool, count))
    if sum(positives[task] for task in tasks) >= minimum:
        break
else:
    raise SystemExit("no test draw with enough verified dangerous messages")
json.dump({"seed": seed, "attempt": attempt, "tasks": tasks,
           "known_dangerous": {task: positives[task] for task in tasks}}, open(path, "w"), indent=2)
print(f"test task pairs: {tasks}")
PY
fi
HOLDOUT_TASKS="$("${UV[@]}" python -c 'import json, sys; print(" ".join(json.load(open(sys.argv[1]))["tasks"]))' "$STATE/test_split.json")"
# shellcheck disable=SC2206  # space-separated list
COMMON=(--state "$STATE" --exclude-after-attack --eval-all-runs --holdout-tasks $HOLDOUT_TASKS
  --validation-tasks "${VALIDATION_TASKS:-2}" --validation-seed "${VALIDATION_SEED:-0}"
  --protect behaviour
  --min-replays "${MIN_REPLAYS:-5}" --max-replays "${MAX_REPLAYS:-10}"
  --zero-safe-at "${ZERO_SAFE_AT:-10}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}")
part=()
[ -n "${2:-}" ] && part=(--part "$2")
versions=()
# shellcheck disable=SC2206
[ -n "${VERSIONS:-}" ] && versions=(--versions $VERSIONS)

accepted_versions() {
  # v1 and every version the loop accepted, in order (the learning curve)
  "${UV[@]}" python -c 'import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
print(" ".join(row["version"] for row in rows if row["accepted"]))' "$STATE/iterations.jsonl"
}

case "${1:-}" in
  run)
    "${EVOLVE[@]}" run "${COMMON[@]}" --verify-baseline \
      --label-budget "${LABEL_BUDGET:-10}" --label-concurrency "$LABEL_CONCURRENCY" \
      --max-iterations "${MAX_ITERATIONS:-18}" --patience "${PATIENCE:-8}"
    ;;
  status) "${EVOLVE[@]}" status "${COMMON[@]}" ;;
  holdout-cost) "${EVOLVE[@]}" holdout "${COMMON[@]}" "${versions[@]}" ;;
  holdout)
    "${EVOLVE[@]}" holdout "${COMMON[@]}" "${versions[@]}" --label \
      --label-concurrency "$LABEL_CONCURRENCY" "${part[@]}"
    ;;
  curve-cost)
    # shellcheck disable=SC2046
    "${EVOLVE[@]}" holdout "${COMMON[@]}" --versions $(accepted_versions)
    ;;
  curve)
    # shellcheck disable=SC2046
    "${EVOLVE[@]}" holdout "${COMMON[@]}" --versions $(accepted_versions) --label \
      --label-concurrency "$LABEL_CONCURRENCY" "${part[@]}"
    ;;
  export) "${UV[@]}" python scripts/self_evolve_attribution.py export-labels ;;
  *) sed -n '2,56p' "$0"; exit 1 ;;
esac
