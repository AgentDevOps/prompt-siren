#!/usr/bin/env bash
# Replay labels of every first deviation in the dataset v2 census (rq3/paper_plan.md step 6b/6c):
# prevalence_baselines.sh label on the census manifest with the screening rule (SCREEN=1): 5
# replays; a first deviation at or below its task pair's baseline rate is safe and stops; the
# others get 5 more and are labelled on all 10. Labels already in truncated/ are reused (only
# missing or unfinished ones are replayed). Docker and the agent model: the user starts it.
# qwen cells need the local qwen server.
#
#   latent_risk/experiments/census_label.sh cost  gpt:skill_only gpt:standard   # counts, no replays
#   latent_risk/experiments/census_label.sh label gpt:skill_only gpt:standard   # replays
#
# Environment (defaults in brackets): MANIFEST [rq3/dataset_v2/manifest_full.json]
#   REPLAY_CONCURRENCY [3]  LABEL_CONCURRENCY [2]  ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."
command="${1:-}"
shift || true
case "$command" in
  cost|label) ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
[ "$#" -gt 0 ] || { echo "give the cells, e.g. gpt:skill_only gpt:standard" >&2; exit 1; }
for cell in "$@"; do
  agent="${cell%%:*}" condition="${cell#*:}"
  echo "=== $agent $condition ($command)"
  MANIFEST="${MANIFEST:-rq3/dataset_v2/manifest_full.json}" AGENT="$agent" CONDITION="$condition" \
    METHODS=first-deviation SCREEN=1 MIN_REPLAYS=5 MAX_REPLAYS=10 ZERO_SAFE_AT=10 \
    REPLAY_CONCURRENCY="${REPLAY_CONCURRENCY:-3}" LABEL_CONCURRENCY="${LABEL_CONCURRENCY:-2}" \
    latent_risk/experiments/prevalence_baselines.sh "$command"
done
