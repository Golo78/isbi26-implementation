#!/usr/bin/env bash
# Train the FP/FN heads fpfn_<k> for all pmri splits (train_fpfn_head.sh), one head index k
# per GPU at a time; each job trains the three splits in sequence.
#   usage: GPUS="2 3 4 6 7" [ITERS="0 1 2 3 4"] [other train_fpfn_head.sh overrides] bash run_fpfn_head_queue.sh
set -uo pipefail

GPUS=(${GPUS:?set GPUS, e.g. "2 3 4"})
ITERS=(${ITERS:-0 1 2 3 4})
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/../../logs"
mkdir -p "$LOGDIR"

run_gpu() {
    local gpu=$1; shift
    for k in "$@"; do
        local log="$LOGDIR/train_fpfn_head_${k}.log"
        echo "[$(date +%H:%M)] GPU $gpu: fpfn_$k -> $log"
        CUDA_DEVICE=$gpu ITERATIONS=$k bash "$HERE/train_fpfn_head.sh" > "$log" 2>&1 \
            || echo "[$(date +%H:%M)]   !! FAILED: fpfn_$k (see $log)"
    done
    echo "[$(date +%H:%M)] GPU $gpu done"
}

for g in "${!GPUS[@]}"; do
    mine=()
    for j in "${!ITERS[@]}"; do (( j % ${#GPUS[@]} == g )) && mine+=("${ITERS[$j]}"); done
    (( ${#mine[@]} )) && run_gpu "${GPUS[$g]}" "${mine[@]}" &
done
wait
echo "[$(date +%H:%M)] FP/FN head training queue done"
