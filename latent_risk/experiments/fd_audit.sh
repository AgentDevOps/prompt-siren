#!/usr/bin/env bash
# Are the extractor's first deviations at the right message? (see fd_audit.py): an auditor told
# the attacker's goal marks each run's first mention of the injected instruction and first uptake
# of it; the report compares them with the first deviation. LLM calls only, no replays.
#
#   latent_risk/experiments/fd_audit.sh audit    # audit every run, then write $OUT/report.md
#   latent_risk/experiments/fd_audit.sh report   # $OUT/report.md from the saved audits
#
# Environment (defaults in brackets): DATASETS [d1 gpt:skill_only gpt:standard qwen:standard]
#   MODEL [gpt-5.6-luna]  CONCURRENCY [8]  OUT [truncated/fd_audit]  ENV_FILE [.env]
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
OUT="${OUT:-truncated/fd_audit}"
# shellcheck disable=SC2206  # space-separated list
ARGS=(--datasets ${DATASETS:-d1 gpt:skill_only gpt:standard qwen:standard}
  --model "${MODEL:-gpt-5.6-luna}" --concurrency "${CONCURRENCY:-8}" --out "$OUT")
PY=("${UV[@]}" python latent_risk/experiments/fd_audit.py)
case "${1:-}" in
  report) "${PY[@]}" report "${ARGS[@]}" ;;
  audit) mkdir -p "$OUT"; "${PY[@]}" audit "${ARGS[@]}" 2>&1 | tee -a "$OUT.log" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
