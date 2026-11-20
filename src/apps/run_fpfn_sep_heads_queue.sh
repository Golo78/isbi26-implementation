#!/usr/bin/env bash
# Separate FP and FN heads (model/fpfn_head.py, count_output fp | fn): for every k one FP-only
# head <TAG>fp_<k> and one FN-only head <TAG>fn_<k>; jobs (fp|fn x k) are dealt round-robin to
# the GPUs, each GPU runs its jobs in sequence.  Default: 3T->1.5T only, otherwise the settings
# of the joint hard head fpfn_<k> (hard counts, one adversary on the head's own count).
#   -> pre-trained/fpfn-heads/*<TAG>fp_<k>_*.ckpt, *<TAG>fn_<k>_*.ckpt; results/fpfn_head/...
#   -> logs/train_<TAG>fp_head_<k>.log, train_<TAG>fn_head_<k>.log
# TTA with the pair:  HEAD_TAG=<TAG>fp HEAD_TAG_FN=<TAG>fn SOURCES=head bash run_fpfn_only_queue.sh
#   usage: GPUS="2 3 4 5 6 7" [TAG=fpfnsep] [ITERS="0 1 2 3 4"] [SPLITS=threet-to-onepointfivet]
#          [N_EPOCHS=400] [LR_MONITOR=val_loss] [other train_fpfn_head.sh overrides] bash run_fpfn_sep_heads_queue.sh
set -uo pipefail

GPUS=(${GPUS:?set GPUS, e.g. "2 3 4"})
TAG=${TAG:-fpfnsep}
ITERS=${ITERS:-"0 1 2 3 4"}
export SPLITS=${SPLITS:-threet-to-onepointfivet}
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/../../logs"
mkdir -p "$LOGDIR"

JOBS=()
for k in $ITERS; do for out in fp fn; do JOBS+=("$out:$k"); done; done

run_gpu() {
    local gpu=$1; shift
    for job in "$@"; do
        local out k
        IFS=: read -r out k <<< "$job"
        local name="${TAG}${out}"
        local log="$LOGDIR/train_${name}_head_${k}.log"
        echo "[$(date +%H:%M)] GPU $gpu: ${name}_$k -> $log"
        CUDA_DEVICE=$gpu ITERATIONS=$k COUNT_OUTPUT=$out NAME_OVERRIDE=$name \
            bash "$HERE/train_fpfn_head.sh" > "$log" 2>&1 \
            || echo "[$(date +%H:%M)]   !! FAILED: ${name}_$k (see $log)"
    done
    echo "[$(date +%H:%M)] GPU $gpu done"
}

for g in "${!GPUS[@]}"; do
    mine=()
    for j in "${!JOBS[@]}"; do (( j % ${#GPUS[@]} == g )) && mine+=("${JOBS[$j]}"); done
    (( ${#mine[@]} )) && run_gpu "${GPUS[$g]}" "${mine[@]}" &
done
wait
echo "[$(date +%H:%M)] separate FP/FN head queue done"
