#!/usr/bin/env bash
# Does evolution transfer with more data (option 1) and the replay-rate fitness (option 2)?
# 2 x 2 x seeds with fd_evolve.py: fitness label AUC vs within-pair rate concordance; training on
# half vs all of the training task pairs (the same held-out pairs, a third of the task pairs).
# Held-out change (final - initial) in label AUC and rate concordance. LLM calls only.
#
#   latent_risk/experiments/fd_evolve_grid.sh warm     # initial instructions on all points (shared)
#   latent_risk/experiments/fd_evolve_grid.sh run      # the grid, one seed's 4 runs in parallel
#   latent_risk/experiments/fd_evolve_grid.sh report   # $OUT/grid_report.md
#
# Environment (defaults in brackets): SEEDS [0 1 2 3]  ROUNDS [4]  TEST_SHARE [0.3333]
#   FRACTIONS [0.5 1]  TARGETS [label rate]  CONCURRENCY [6]  OUT [truncated/fd_evolve_grid]
#   ENV_FILE [.env]
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
OUT="${OUT:-truncated/fd_evolve_grid}"
TEST_SHARE="${TEST_SHARE:-0.3333}"
PY=("${UV[@]}" python latent_risk/experiments/fd_evolve.py)
COMMON=(--test-share "$TEST_SHARE" --concurrency "${CONCURRENCY:-6}" --cache "$OUT/scores")
case "${1:-}" in
  warm)
    mkdir -p "$OUT"
    "${PY[@]}" evolve "${COMMON[@]}" --rounds 0 --out "$OUT/warm" 2>&1 | tee -a "$OUT.log" ;;
  run)
    mkdir -p "$OUT"
    for seed in ${SEEDS:-0 1 2 3}; do
      for target in ${TARGETS:-label rate}; do
        for fraction in ${FRACTIONS:-0.5 1}; do
          name="${target}_f${fraction}_s${seed}"
          [ -f "$OUT/$name/summary.json" ] && continue
          "${PY[@]}" evolve "${COMMON[@]}" --rounds "${ROUNDS:-4}" --seed "$seed" \
            --target "$target" --train-fraction "$fraction" --out "$OUT/$name" \
            >"$OUT/$name.log" 2>&1 &
        done
      done
      wait
      echo "seed $seed done"
    done ;;
  report) "${PY[@]}" grid-report --out "$OUT" ;;
  *) sed -n '2,/^set -euo/{/^set -euo/!p}' "$0"; exit 1 ;;
esac
