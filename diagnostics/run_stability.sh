#!/usr/bin/env bash
# Runs the stability_vs_length.py grid: L=8 and L=16 first (cheap), then the L=100 reference runs.
# Per-run logs: diagnostics/results/stability_L*_lr*_clip*.log; console log: logs/diagnostics/ (gitignored).
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash diagnostics/run_stability.sh [--lengths 8 16] [--skip-existing]
# With no arguments it runs the whole grid in two phases.
set -u
cd "$(dirname "$0")"
PY=${PY:-/home/a84460786/EdgeAlign/.venv/bin/python}
LOG_DIR=../logs/diagnostics
mkdir -p "$LOG_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {  # run <log-name> [args...]
    local name=$1; shift
    echo "=== $name ($(date -u +%H:%M:%S) UTC)"
    "$PY" stability_vs_length.py "$@" 2>&1 | grep --line-buffered -v "Loading weights" > "$LOG_DIR/$name.log"
    echo "    exit=${PIPESTATUS[0]}"
}

if [ $# -gt 0 ]; then
    run stability_custom "$@"
else
    run stability_short --lengths 8 16
    run stability_L100 --lengths 100
fi
echo "=== all done ($(date -u +%H:%M:%S) UTC)"
