#!/usr/bin/env bash
# Regenerates every figure of the paper from experiments/data/ into figures/.
# Usage: ./reproduce.sh            (about 10-20 minutes on a laptop; all CPU, no network)
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/arbitrage-matplotlib}"
mkdir -p "$MPLCONFIGDIR"
PY=${PYTHON:-python}
# The documented invocation passes a repository-relative interpreter
# (`PYTHON=.venv/bin/python`).  Resolve path-like values before changing into
# experiments/ below; bare command names such as `python3` remain PATH lookups.
if [[ "$PY" == */* && "$PY" != /* ]]; then
  PY="$(pwd)/$PY"
fi
mkdir -p figures plots/fig1_data plots/fig_order_data plots/fig_passk_data plots/gap_cache experiments/val_cache
cd experiments
echo "== Figure 1: cost to reach a target solve rate per system (Terminal-Bench 2.0)"
"$PY" make_cost_bars.py
echo "== Figures 2 and 6: cross-validated arbitrage frontiers"
"$PY" cv.py --paper
"$PY" ../plots/fig1.py
echo "== Figures 3 and 7: revenue shares (per-problem replay of the cross-validation)"
"$PY" fig_appendix.py
echo "== Figures 4 and 8: deployment order"
"$PY" cv_order.py
"$PY" ../plots/fig_order.py
echo "== Figures 5 and 10: arbitrage with estimated utility curves"
"$PY" validate_calibration.py
"$PY" make_fig_passk_data.py monkey_math monkey_codecontests resmat2_aime2024 resmat2_aime2025 resmat2_gmmlu resmat2_mmlupro
"$PY" ../plots/fig_calib.py
echo "== Figure 9: estimation error of the plug-in vs our estimator"
"$PY" ../plots/fig_gap_dist.py
cd ..
echo "== comparing with the records behind the submitted figures"
"$PY" check_reproduction.py
echo "done: see figures/"
