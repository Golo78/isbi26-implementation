#!/usr/bin/env bash
# Experiment 8 (FP/FN only, no mu-hat) for all pmri splits, 100 steps, pre-norm predictors.
# Jobs (source x ckpt) are dealt round-robin to the GPUs; each GPU runs its jobs in sequence,
# each job runs the three splits in sequence (~1 h per job).
#   -> results/latent_tta_fpfnonly_<source>/
# HEAD_TAG=<tag> uses the heads <tag>_<k> instead of fpfn_<k> (source=head only)
# HEAD_TAG_FN=<tag> additionally: separate heads, <HEAD_TAG>_<k> = FP-only, <HEAD_TAG_FN>_<k> = FN-only
#   -> results/latent_tta_fpfnonly_head_<tag>/, logs tta100_fpfnonly_head_<tag>_cleanHalf_<k>.log
#   usage: GPUS="2 3 4 6 7" [SOURCES="oracle head"] [CKPTS="0 1 2 3 4"] [FPFN_FN_WEIGHT=1.0] [HEAD_TAG=fpfnsplit] bash run_fpfn_only_queue.sh
set -uo pipefail

GPUS=(${GPUS:?set GPUS, e.g. "2 3 4"})
SOURCES=${SOURCES:-"oracle head"}
CKPTS=${CKPTS:-"0 1 2 3 4"}
FPFN_FN_WEIGHT=${FPFN_FN_WEIGHT:-1.0}
HEAD_TAG=${HEAD_TAG:-}
HEAD_TAG_FN=${HEAD_TAG_FN:-}
if [[ $FPFN_FN_WEIGHT != 1.0 && $FPFN_FN_WEIGHT != 1 ]]; then FNTAG=_fn$FPFN_FN_WEIGHT; else FNTAG=""; fi
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/../../logs"
mkdir -p "$LOGDIR"

JOBS=()
for S in $SOURCES; do for I in $CKPTS; do JOBS+=("$S:$I"); done; done

run_gpu() {
    local gpu=$1; shift
    for job in "$@"; do
        local s i
        IFS=: read -r s i <<< "$job"
        local tag="" head=null head_fn=null
        if [[ $s == head && -n $HEAD_TAG ]]; then tag="_$HEAD_TAG"; head="${HEAD_TAG}_$i"; fi
        if [[ $s == head && -n $HEAD_TAG_FN ]]; then tag="${tag}_$HEAD_TAG_FN"; head_fn="${HEAD_TAG_FN}_$i"; fi
        local log="$LOGDIR/tta100_fpfnonly_${s}${tag}${FNTAG}_cleanHalf_${i}.log"
        echo "[$(date +%H:%M)] GPU $gpu: fpfnonly $s$tag cleanHalf_$i -> $log"
        CUDA_DEVICE=$gpu MAX_STEPS=100 CKPT_NAME="cleanHalf_$i" \
        FPFN_SOURCE=$s FPFN_FN_WEIGHT=$FPFN_FN_WEIGHT FPFN_HEAD_NAME=$head FPFN_HEAD_NAME_FN=$head_fn \
        RESULT_DIR=latent_tta_fpfnonly_${s}${tag}${FNTAG} \
            bash "$HERE/exp8_fpfn_only.sh" > "$log" 2>&1 \
            || echo "[$(date +%H:%M)]   !! FAILED: fpfnonly $s cleanHalf_$i (see $log)"
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
echo "[$(date +%H:%M)] fpfnonly queue done"
