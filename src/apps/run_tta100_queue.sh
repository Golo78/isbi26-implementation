#!/usr/bin/env bash
# Sequential queue of 100-step latent TTA runs on ONE GPU, for the scenarios
#   postnorm           post-norm predictor, sites [0,17]
#   postnorm_logits    post-norm predictor, sites [0,17] + logits
#   prenorm            pre-norm predictor,  sites [0,17]
#   prenorm_logits     pre-norm predictor,  sites [0,17] + logits
# over all three pmri splits, for each predictor in CKPTS.
#
# Waits first until no other `eval_latent_tta.sh` chain is running on the same GPU:
# two processes on one GPU were measured at ~2x the per-batch time each, i.e. no gain in
# throughput, so one-at-a-time delivers finished results earlier.
#
#   usage: GPU=6 CKPTS="1 3" [SCENARIOS="prenorm_logits"] bash run_tta100_queue.sh
set -uo pipefail

GPU=${GPU:?set GPU}
CKPTS=${CKPTS:?set CKPTS, e.g. "1 3"}
SCENARIOS=${SCENARIOS:-"postnorm postnorm_logits prenorm prenorm_logits"}
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="$HERE/../../logs"
mkdir -p "$LOGDIR"

chains_on_gpu() {
    for p in $(pgrep -f "bash eval_latent_tta.sh"); do
        [ "$p" = "$$" ] && continue
        tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -qx "CUDA_DEVICE=$GPU" && echo "$p"
    done
}

while [ -n "$(chains_on_gpu)" ]; do
    echo "[$(date +%H:%M)] waiting for running chain(s) on GPU $GPU: $(chains_on_gpu | tr '\n' ' ')"
    sleep 120
done

cd "$HERE"
for I in $CKPTS; do
    for SCEN in $SCENARIOS; do
        case $SCEN in
            postnorm)          SUBDIR=post-norm_with_logits; LOGITS=False; RDIR=latent_tta_postnorm ;;
            postnorm_logits)   SUBDIR=post-norm_with_logits; LOGITS=True;  RDIR=latent_tta_postnorm_logits ;;
            prenorm)           SUBDIR=pre_norm_with_logits;  LOGITS=False; RDIR=latent_tta_prenorm ;;
            prenorm_logits)    SUBDIR=pre_norm_with_logits;  LOGITS=True;  RDIR=latent_tta_prenorm_logits ;;
            *) echo "unknown scenario '$SCEN'" >&2; exit 2 ;;
        esac
        LOG="$LOGDIR/tta100_${SCEN}_cleanHalf_${I}.log"
        echo "[$(date +%H:%M)] GPU $GPU: $SCEN cleanHalf_$I -> $LOG"
        CUDA_DEVICE=$GPU MAX_STEPS=100 OBJECTIVE=mu PLACEMENT=post KAPPA_WEIGHT=0.0 \
        CKPT_NAME="cleanHalf_$I" CKPT_SUBDIR=$SUBDIR ADAPT_LOGITS=$LOGITS RESULT_DIR=$RDIR \
            bash eval_latent_tta.sh > "$LOG" 2>&1 \
            || echo "[$(date +%H:%M)]   !! FAILED: $SCEN cleanHalf_$I (see $LOG)"
    done
done
echo "[$(date +%H:%M)] queue on GPU $GPU done"
