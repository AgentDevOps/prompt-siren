#!/usr/bin/env bash
# Claim 2 (dangerous messages are concentrated in certain task pairs): equal-effort random audit.
#
#   latent_risk/experiments/claim2_audit.sh cost            # sample size and replays still needed
#   latent_risk/experiments/claim2_audit.sh label [K/N]     # replay-verify the sample (optional share)
#   latent_risk/experiments/claim2_audit.sh report          # rq3/claim2/<POSITION>/report.md
#
# POSITION: which message per run is audited.
#   first-chain  the first agent message of the run's attack chain (the agent's first engagement
#                with the injection); default
#   any          a random agent message before the attack action (the first pilot)
#
# Pilot (defaults): 3 runs per task pair, 5 replays per message. The sample is a seeded order per
# task pair, so a larger PER_PAIR extends it and a larger MAX_REPLAYS tops up the same messages:
#   PER_PAIR=6 MAX_REPLAYS=10 ZERO_SAFE_AT=10 latent_risk/experiments/claim2_audit.sh label
#
# Environment (defaults in brackets): POSITION [first-chain]  PER_PAIR [3]  MAX_REPLAYS [5]
#   ZERO_SAFE_AT [5]  SEED [0]
#   REPLAY_CONCURRENCY [3]  LABEL_CONCURRENCY [1]  ENV_FILE [.env]
set -euo pipefail
cd "$(dirname "$0")/../.."
ENV_FILE="${ENV_FILE:-.env}"
UV=(uv run)
[ -f "$ENV_FILE" ] && UV+=(--env-file "$ENV_FILE")
POSITION="${POSITION:-first-chain}"
ARGS=(--position "$POSITION" --out "rq3/claim2/$POSITION"
  --per-pair "${PER_PAIR:-3}" --max-replays "${MAX_REPLAYS:-5}"
  --zero-safe-at "${ZERO_SAFE_AT:-5}" --seed "${SEED:-0}"
  --replay-concurrency "${REPLAY_CONCURRENCY:-3}" --label-concurrency "${LABEL_CONCURRENCY:-1}")
case "${1:-}" in
  cost) "${UV[@]}" python latent_risk/experiments/claim2_audit.py cost "${ARGS[@]}" ;;
  label)
    part=()
    [ -n "${2:-}" ] && part=(--part "$2")
    "${UV[@]}" python latent_risk/experiments/claim2_audit.py label "${ARGS[@]}" "${part[@]}"
    ;;
  report) "${UV[@]}" python latent_risk/experiments/claim2_audit.py report "${ARGS[@]}" ;;
  *) sed -n '2,19p' "$0"; exit 1 ;;
esac
