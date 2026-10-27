#!/usr/bin/env bash
# Experiment 1, step 1: independent confirmation of the verified dangerous messages.
#
#   experiments/checkpoint_confirm.sh run [K/N]   # new cut + fixed fresh replays per message
#   experiments/checkpoint_confirm.sh report      # truncated/checkpoint_confirm/report.md
#
# The 15 targets are frozen in truncated/checkpoint_confirm/targets.json on the first call. The
# fresh replays are stored there, apart from the discovery replays; a restart tops up unfinished
# messages to the fixed count.
#
# Environment (defaults in brackets): REPLAYS [10]  REPLAY_CONCURRENCY [3]  LABEL_CONCURRENCY [1]
#   ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/.."
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
ARGS=(--replays "${REPLAYS:-10}" --replay-concurrency "${REPLAY_CONCURRENCY:-3}"
  --label-concurrency "${LABEL_CONCURRENCY:-1}")
case "${1:-}" in
  run)
    part=()
    [ -n "${2:-}" ] && part=(--part "$2")
    "${UV[@]}" python experiments/checkpoint_confirm.py run "${ARGS[@]}" "${part[@]}"
    ;;
  report) "${UV[@]}" python experiments/checkpoint_confirm.py report "${ARGS[@]}" ;;
  *) sed -n '2,12p' "$0"; exit 1 ;;
esac
