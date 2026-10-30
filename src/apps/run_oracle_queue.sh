#!/usr/bin/env bash
# Oracle upper bound (ascent on the TRUE soft Dice) for all pmri splits, 100 steps, in the
# two adaptation configurations of the 2x2 design:
#   sites    sites [0,17]           -> results/latent_tta_oracle/
#   logits   sites [0,17] + logits  -> results/latent_tta_oracle_logits/
# The objective does not use mu-hat, so the predictor only provides the logged mu/kappa
# trajectories; the pre-norm predictors are used (same checkpoints as latent_tta_prenorm*).
#
# Jobs (config x ckpt) are dealt round-robin to the GPUs; each GPU runs its jobs in sequence.
#   usage: GPUS="2 3 4 6 7" [CKPTS="0 1 2 3 4"] [CONFIGS="sites logits"] bash run_oracle_queue.sh
set -uo pipefail

GPUS=(${GPUS:?set GPUS, e.g. "2 3 4"})
CKPTS=${CKPTS:-"0 1 2 3 4"}
CONFIGS=${CONFIGS:-"sites logits"}
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/../../logs"
mkdir -p "$LOGDIR"

JOBS=()
for C in $CONFIGS; do for I in $CKPTS; do JOBS+=("$C:$I"); done; done

run_gpu() {
    local gpu=$1; shift
    for job in "$@"; do
        local cfg=${job%%:*} i=${job##*:} logits rdir
        case $cfg in
            sites)  logits=False; rdir=latent_tta_oracle ;;
            logits) logits=True;  rdir=latent_tta_oracle_logits ;;
            *) echo "unknown config '$cfg'" >&2; return 2 ;;
        esac
        local log="$LOGDIR/tta100_oracle_${cfg}_cleanHalf_${i}.log"
        echo "[$(date +%H:%M)] GPU $gpu: oracle $cfg cleanHalf_$i -> $log"
        CUDA_DEVICE=$gpu MAX_STEPS=100 CKPT_NAME="cleanHalf_$i" CKPT_SUBDIR=pre_norm_with_logits \
        ADAPT_LOGITS=$logits RESULT_DIR=$rdir \
            bash "$HERE/exp1_oracle_dice.sh" > "$log" 2>&1 \
            || echo "[$(date +%H:%M)]   !! FAILED: oracle $cfg cleanHalf_$i (see $log)"
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
echo "[$(date +%H:%M)] oracle queue done"
