#!/usr/bin/env bash
# FP/FN oracle (experiment 6) for all pmri splits, 100 steps, pre-norm predictors.
#   mu-hat + descent on the true FP/FN, FPFN_WEIGHT in $WEIGHTS -> results/latent_tta_mu_fpfn_w<w>/
#
# Jobs (weight x ckpt) are dealt round-robin to the GPUs; each GPU runs its jobs in sequence.
#   FPFN_FORM=joint descends log(1 + FP + FN) instead -> results/latent_tta_mu_fpfn_joint_w<w>/
#   FPFN_FN_WEIGHT=a weights FN inside the term -> results/latent_tta_mu_fpfn[_joint]_fn<a>_w<w>/
#   usage: GPUS="2 3 4 6 7" [CKPTS="0 1 2 3 4"] [WEIGHTS="0.1 1.0"] [FPFN_FORM=separate] [FPFN_FN_WEIGHT=1.0] bash run_fpfn_oracle_queue.sh
set -uo pipefail

GPUS=(${GPUS:?set GPUS, e.g. "2 3 4"})
CKPTS=${CKPTS:-"0 1 2 3 4"}
WEIGHTS=${WEIGHTS:-"0.1 1.0"}
FPFN_FORM=${FPFN_FORM:-separate}
FPFN_FN_WEIGHT=${FPFN_FN_WEIGHT:-1.0}
if [[ $FPFN_FORM == separate ]]; then TAG=fpfn; else TAG=fpfn_$FPFN_FORM; fi
if [[ $FPFN_FN_WEIGHT != 1.0 && $FPFN_FN_WEIGHT != 1 ]]; then TAG=${TAG}_fn$FPFN_FN_WEIGHT; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/../../logs"
mkdir -p "$LOGDIR"

JOBS=()
for W in $WEIGHTS; do for I in $CKPTS; do JOBS+=("$W:$I"); done; done

run_gpu() {
    local gpu=$1; shift
    for job in "$@"; do
        local w i
        IFS=: read -r w i <<< "$job"
        local log="$LOGDIR/tta100_${TAG}_w${w}_cleanHalf_${i}.log"
        echo "[$(date +%H:%M)] GPU $gpu: $TAG w=$w cleanHalf_$i -> $log"
        CUDA_DEVICE=$gpu MAX_STEPS=100 CKPT_NAME="cleanHalf_$i" \
        FPFN_WEIGHT=$w FPFN_FORM=$FPFN_FORM FPFN_FN_WEIGHT=$FPFN_FN_WEIGHT \
            bash "$HERE/exp6_fpfn_oracle.sh" > "$log" 2>&1 \
            || echo "[$(date +%H:%M)]   !! FAILED: $TAG w=$w cleanHalf_$i (see $log)"
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
echo "[$(date +%H:%M)] $TAG oracle queue done"
