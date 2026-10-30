#!/usr/bin/env bash
set -euo pipefail

# Latent-space test-time adaptation evaluation.
#
# Environment overrides
#   CUDA_DEVICE     GPU id (default 0)
#   DATASETS        Space-separated datasets (default: "pmri")
#   SPLITS          Space-separated split tags (overrides per-dataset defaults)
#   CKPT_NAME       Substring selecting the score predictor (e.g. cleanHalf_0)
#   CKPT_SUBDIR     Subdirectory under pre-trained/score-predictors
#                   (use score-predictors-using-logits for the logits variant)
#   ADAPT_SITES     Python list of adn.N indices, e.g. "[0,17]"
#   MAX_STEPS       Ascent steps (default 20)
#   STOP_RULE       fixed-steps | delta-mu-target | fg-drift
#   GRANULARITY     batch | domain
#   OBJECTIVE       mu | oracle-dice   (oracle-dice uses labels: upper bound, not a method)
#   PLACEMENT       post | pre         (affine after / before the swivel InstanceNorm)
#   KAPPA_WEIGHT    weight of the simultaneous uncertainty descent (0 = off)
#   KAPPA_OBJECTIVE sigma | log-kappa
#   ADAPT_LOGITS    True | False       (additional affine on the UNet logits)
#   LOGIT_LR        Adam lr of the logit affine (default: null = same lr)
#   N_EXAMPLES      images per domain snapshotted + plotted (default 6)
#   MAX_BATCHES     only the first N batches per domain (quick checks; default: all)
#
# The score predictor's hook_placement (pre/post-norm) is read from the checkpoint.
#   RESULT_DIR      subdirectory under results/ (keep experiments apart!)

CUDA_DEVICE=${CUDA_DEVICE:-0}
DATASETS=${DATASETS:-"pmri"}
CKPT_NAME=${CKPT_NAME:-cleanHalf_0}
CKPT_SUBDIR=${CKPT_SUBDIR:-pre_norm_with_logits}
ADAPT_SITES=${ADAPT_SITES:-"[0,17]"}
MAX_STEPS=${MAX_STEPS:-20}
STOP_RULE=${STOP_RULE:-fixed-steps}
GRANULARITY=${GRANULARITY:-batch}
BATCH_SIZE=${BATCH_SIZE:-8}
OBJECTIVE=${OBJECTIVE:-mu}
PLACEMENT=${PLACEMENT:-post}
KAPPA_WEIGHT=${KAPPA_WEIGHT:-0.0}
KAPPA_OBJECTIVE=${KAPPA_OBJECTIVE:-sigma}
RESULT_DIR=${RESULT_DIR:-latent_tta}
INCLUDE_TRAIN=${INCLUDE_TRAIN:-False}
ADAPT_LOGITS=${ADAPT_LOGITS:-False}
LOGIT_LR=${LOGIT_LR:-null}
N_EXAMPLES=${N_EXAMPLES:-6}
MAX_BATCHES=${MAX_BATCHES:-null}

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
        echo "Latent TTA: dataset=$DATA split=$SPLIT sites=$ADAPT_SITES steps=$MAX_STEPS"
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_latent_tta.py \
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
            +tta=latent_affine \
            ++tta.adapt_sites="$ADAPT_SITES" \
            ++tta.max_steps="$MAX_STEPS" \
            ++tta.stop_rule="$STOP_RULE" \
            ++tta.granularity="$GRANULARITY" \
            ++tta.objective="$OBJECTIVE" \
            ++tta.placement="$PLACEMENT" \
            ++tta.kappa_weight="$KAPPA_WEIGHT" \
            ++tta.kappa_objective="$KAPPA_OBJECTIVE" \
            ++tta.adapt_logits="$ADAPT_LOGITS" \
            ++tta.logit_lr="$LOGIT_LR" \
            ++tta.n_examples="$N_EXAMPLES" \
            ++max_batches="$MAX_BATCHES" \
            ++ckpt_name="$CKPT_NAME" \
            ++ckpt_subdir="$CKPT_SUBDIR" \
            ++include_train="$INCLUDE_TRAIN"
        echo "Completed $DATA/$SPLIT"
    done
done
echo "All latent TTA evaluations completed!"
