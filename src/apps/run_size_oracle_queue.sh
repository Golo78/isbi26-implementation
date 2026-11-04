#!/usr/bin/env bash
# Size oracle (experiment 5) for all pmri splits, 100 steps, pre-norm predictors.
#   O3: mu-hat + size anchor, SIZE_WEIGHT in $WEIGHTS, no noise   -> results/latent_tta_mu_size_w<w>/
#   C : the same with noisy FP/FN, SIZE_NOISE in $NOISES          -> results/latent_tta_mu_size_w<w>_noise<n>/
# Leave NOISES empty for O3 only; for C set WEIGHTS to the best weight from O3.
#
# Jobs (weight x noise x ckpt) are dealt round-robin to the GPUs; each GPU runs its jobs in sequence.
#   usage: GPUS="2 3 4 6 7" [CKPTS="0 1 2 3 4"] [WEIGHTS="0.1 1.0"] [NOISES="0.0"] bash run_size_oracle_queue.sh
#   e.g.   GPUS="2 3" CKPTS="0 1" WEIGHTS="1.0" NOISES="0.1 0.2 0.3" bash run_size_oracle_queue.sh
set -uo pipefail

GPUS=(${GPUS:?set GPUS, e.g. "2 3 4"})
CKPTS=${CKPTS:-"0 1 2 3 4"}
WEIGHTS=${WEIGHTS:-"0.1 1.0"}
NOISES=${NOISES:-"0.0"}
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/../../logs"
mkdir -p "$LOGDIR"

JOBS=()
for W in $WEIGHTS; do for N in $NOISES; do for I in $CKPTS; do JOBS+=("$W:$N:$I"); done; done; done

run_gpu() {
    local gpu=$1; shift
    for job in "$@"; do
        local w n i
        IFS=: read -r w n i <<< "$job"
        local log="$LOGDIR/tta100_size_w${w}_noise${n}_cleanHalf_${i}.log"
        echo "[$(date +%H:%M)] GPU $gpu: size w=$w noise=$n cleanHalf_$i -> $log"
        CUDA_DEVICE=$gpu MAX_STEPS=100 CKPT_NAME="cleanHalf_$i" \
        SIZE_WEIGHT=$w SIZE_NOISE=$n \
            bash "$HERE/exp5_size_oracle.sh" > "$log" 2>&1 \
            || echo "[$(date +%H:%M)]   !! FAILED: size w=$w noise=$n cleanHalf_$i (see $log)"
    done
    echo "[$(date +%H:%M)] GPU $gpu done"
}

cd "$HERE"
for g in "${!GPUS[@]}"; do
    mine=()
    for j in "${!JOBS[@]}"; do (( j % ${#GPUS[@]} == g )) && mine+=("${JOBS[$j]}"); done
    run_gpu "${GPUS[$g]}" "${mine[@]}" &
done
wait
echo "[$(date +%H:%M)] size oracle queue done"
