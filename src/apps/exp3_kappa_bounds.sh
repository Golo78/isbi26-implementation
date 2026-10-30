#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 3a -- kappa credible bounds of the CURRENT predictions (no adaptation).
#
# Runs the score predictor once per test domain and reports, per nominal level, the
# empirical coverage and width of the Beta credible interval around mu-hat, the PIT mean,
# and the mu-hat / true-Dice correlation.  This is the "let us first look at the kappa
# confidence bounds" step that precedes the uncertainty descent (exp3_kappa_descent.sh).
#
# The score predictor's hook_placement (pre/post-norm) is read from the checkpoint.
# Writes to results/kappa_bounds/<ckpt>/kappa_bounds.pkl.
#
#   usage: [SPLITS=...] [CKPT_NAME=cleanHalf_0] bash exp3_kappa_bounds.sh

CUDA_DEVICE=${CUDA_DEVICE:-0}
DATASETS=${DATASETS:-"pmri"}
CKPT_NAME=${CKPT_NAME:-cleanHalf_0}
CKPT_SUBDIR=${CKPT_SUBDIR:-pre_norm_with_logits}
BATCH_SIZE=${BATCH_SIZE:-8}
RESULT_DIR=${RESULT_DIR:-kappa_bounds}
INCLUDE_TRAIN=${INCLUDE_TRAIN:-False}

for DATA in ${DATASETS}; do
    if [ "$DATA" = 'mnmv2' ]; then
        UNET_OUT_DIM=4; NUM_CLASSES=4
        DEFAULT_SPLITS=("scanner-symphonytim" "pathology-norm-vs-fall-scanners-all")
    elif [ "$DATA" = 'pmri' ]; then
        UNET_OUT_DIM=1; NUM_CLASSES=2
        DEFAULT_SPLITS=("threet-to-onepointfivet" "promise12" "onepointfivet-to-threet")
    else
        echo "Unknown dataset '$DATA'" >&2; exit 2
    fi

    if [ -n "${SPLITS:-}" ]; then
        # shellcheck disable=SC2206
        TARGET_SPLITS=(${SPLITS})
    else
        TARGET_SPLITS=("${DEFAULT_SPLITS[@]}")
    fi

    for SPLIT in "${TARGET_SPLITS[@]}"; do
        echo "Kappa bounds: dataset=$DATA split=$SPLIT"
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_kappa_bounds.py \
            +data="$DATA" \
            ++data.split="$SPLIT" \
            ++data.non_empty_target=True \
            ++data.batch_size="$BATCH_SIZE" \
            +unet=monai_unet \
            ++unet.out_channels="$UNET_OUT_DIM" \
            +model=score_predictor \
            ++model.num_classes="$NUM_CLASSES" \
            ++model.swivels='first-embedding-last' \
            +trainer=score_predictor_trainer \
            ++trainer.result_dir="$RESULT_DIR" \
            ++ckpt_name="$CKPT_NAME" \
            ++ckpt_subdir="$CKPT_SUBDIR" \
            ++include_train="$INCLUDE_TRAIN"
        echo "Completed $DATA/$SPLIT"
    done
done
echo "All kappa bound evaluations completed!"
