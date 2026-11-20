#!/usr/bin/env bash
# Experiment 9 (pseudo-label oracle) sweep: jobs (mode:level x ckpt) are dealt round-robin
# to the GPUs; each GPU runs its jobs in sequence.  Default: 3T->1.5T only.
#   -> results/latent_tta_pseudo_<mode><level>/, logs/tta100_pseudo_<mode><level>_cleanHalf_<k>.log
#   usage: GPUS="2 3 4 6 7" [LEVELS="recall:1.0 recall:0.75 recall:0.5 recall:0.25 recall:0.0 morph:3 morph:-3"]
#          [CKPTS="0 1 2 3 4"] [SPLITS=threet-to-onepointfivet] bash run_pseudo_oracle_queue.sh
set -uo pipefail

GPUS=(${GPUS:?set GPUS, e.g. "2 3 4"})
LEVELS=${LEVELS:-"recall:1.0 recall:0.75 recall:0.5 recall:0.25 recall:0.0 morph:3 morph:-3"}
CKPTS=${CKPTS:-"0 1 2 3 4"}
export SPLITS=${SPLITS:-threet-to-onepointfivet}
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/../../logs"
mkdir -p "$LOGDIR"

JOBS=()
for L in $LEVELS; do for I in $CKPTS; do JOBS+=("$L:$I"); done; done

run_gpu() {
    local gpu=$1; shift
    for job in "$@"; do
        local m l i
        IFS=: read -r m l i <<< "$job"
        local log="$LOGDIR/tta100_pseudo_${m}${l}_cleanHalf_${i}.log"
        echo "[$(date +%H:%M)] GPU $gpu: pseudo $m $l cleanHalf_$i -> $log"
        CUDA_DEVICE=$gpu MAX_STEPS=100 CKPT_NAME="cleanHalf_$i" \
        PSEUDO_MODE=$m PSEUDO_LEVEL=$l \
            bash "$HERE/exp9_pseudo_oracle.sh" > "$log" 2>&1 \
            || echo "[$(date +%H:%M)]   !! FAILED: pseudo $m $l cleanHalf_$i (see $log)"
    done
    echo "[$(date +%H:%M)] GPU $gpu done"
}

cd "$HERE"
for g in "${!GPUS[@]}"; do
    mine=()
    for j in "${!JOBS[@]}"; do (( j % ${#GPUS[@]} == g )) && mine+=("${JOBS[$j]}"); done
    (( ${#mine[@]} )) && run_gpu "${GPUS[$g]}" "${mine[@]}" &
done
wait
echo "[$(date +%H:%M)] pseudo queue done"
