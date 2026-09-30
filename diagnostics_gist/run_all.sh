#!/usr/bin/env bash
# Runs the gist-health tests in order on one GPU. JSON/PNG results -> diagnostics_gist/results/,
# console logs -> diagnostics_gist/results/logs/.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash diagnostics_gist/run_all.sh [t1_swap t2_variation ...]
set -u
cd "$(dirname "$0")"
PY=${PY:-/data/a84460786/venvs/testfolder/bin/python}
LOG_DIR=results/logs
mkdir -p "$LOG_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TODO=${*:-t1_swap t2_variation t3_rollout t4_baselines t5_sensitivity t6_attention t7_norms t8_steer}
for t in $TODO; do
    echo "=== $t ($(date -u +%H:%M:%S) UTC)"
    "$PY" "$t.py" 2>&1 | grep --line-buffered -v "Loading weights" > "$LOG_DIR/$t.log"
    echo "    exit=${PIPESTATUS[0]}"
done
echo "=== all done ($(date -u +%H:%M:%S) UTC)"
