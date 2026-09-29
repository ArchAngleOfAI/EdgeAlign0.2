#!/usr/bin/env bash
# Runs every diagnostic sequentially on one GPU; logs go to logs/diagnostics/ (gitignored).
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash diagnostics/run_all.sh [script ...]
set -u
cd "$(dirname "$0")"
PY=${PY:-/home/a84460786/EdgeAlign/.venv/bin/python}
LOG_DIR=../logs/diagnostics
mkdir -p "$LOG_DIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {  # run <log-name> <script> [args...]
    local name=$1; shift
    echo "=== $name ($(date -u +%H:%M:%S) UTC)"
    "$PY" "$@" 2>&1 | grep --line-buffered -v "Loading weights" > "$LOG_DIR/$name.log"
    echo "    exit=${PIPESTATUS[0]}"
}

if [ $# -gt 0 ]; then TODO="$*"; else
    TODO="check_gradients check_gradfn check_weight_identity rollout_length_gradient_decay naive_baseline_kd lr_sanity_check capacity_ablation_3layers capacity_ablation_1layer"
fi
for t in $TODO; do
    case $t in
        capacity_ablation_3layers) run "$t" capacity_ablation.py --num-layers 3 ;;
        capacity_ablation_1layer)  run "$t" capacity_ablation.py --num-layers 1 ;;
        *) run "$t" "$t.py" ;;
    esac
done
echo "=== all done ($(date -u +%H:%M:%S) UTC)"
