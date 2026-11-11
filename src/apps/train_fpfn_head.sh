#!/usr/bin/env bash
set -euo pipefail
# Train + evaluate the FP/FN head (model/fpfn_head.py) for every pmri split.
#
# Same data, U-Net checkpoints, batch size and epoch budget as the score predictors
# (train_score_predictor.sh), so head k can be paired with score predictor cleanHalf_k.
# Checkpoints: pre-trained/fpfn-heads/pmri_<split>_fpfn_predictor-adversarial-location_mse-fpfn_<k>_<date>_val-*.ckpt
# Evaluation:  results/fpfn_head/<ckpt name>/eval_data.pkl
#
# Environment overrides
#   CUDA_DEVICE       GPU id (default 0)
#   SPLITS            space-separated pmri splits (default: all three)
#   ITERATIONS        head indices k (default "0 1 2 3 4")
#   N_EPOCHS          max epochs (default 400) | N_BATCHES train batches per epoch (default 100)
#   CKPT_START_EPOCH  first epoch that may be checkpointed (default 150)
#   PERTURBATION      predictor | none
#   ADV_TARGET_DICE_DROP  target drop of the true hard Dice per adversarial step (default 0.1)
#   COUNT_TARGET      hard (thresholded mask, default) | soft (probabilities)
#   ADV_DOWN          True: second adversary in the TTA direction (lowers the predicted error)
#   ADV_DOWN_TARGET_LOG_DROP  its target drop of the predicted log(1+FP+FN) per step (default 0.5)
#   REGRESSION_LOSS   mse | huber
#   LOGGING           True | False (wandb)
#   NAME_OVERRIDE     checkpoint name prefix (default fpfn; e.g. fpfnsoft for COUNT_TARGET=soft)

CUDA_DEVICE=${CUDA_DEVICE:-0}
SPLITS=${SPLITS:-"threet-to-onepointfivet promise12 onepointfivet-to-threet"}
ITERATIONS=${ITERATIONS:-"0 1 2 3 4"}
N_EPOCHS=${N_EPOCHS:-400}
N_BATCHES=${N_BATCHES:-100}
CKPT_START_EPOCH=${CKPT_START_EPOCH:-150}
PERTURBATION=${PERTURBATION:-predictor}
ADVERSARIAL_SKIP_EPOCHS=${ADVERSARIAL_SKIP_EPOCHS:-40}
ADVERSARIAL_WARMUP_EPOCHS=${ADVERSARIAL_WARMUP_EPOCHS:-30}
ADV_TARGET_DICE_DROP=${ADV_TARGET_DICE_DROP:-0.1}
COUNT_TARGET=${COUNT_TARGET:-hard}
ADV_DOWN=${ADV_DOWN:-False}
ADV_DOWN_TARGET_LOG_DROP=${ADV_DOWN_TARGET_LOG_DROP:-0.5}
REGRESSION_LOSS=${REGRESSION_LOSS:-mse}
LOGGING=${LOGGING:-True}
TRAIN=${TRAIN:-True}
TEST=${TEST:-True}
BATCH_SIZE=${BATCH_SIZE:-30}
CKPT_DIR=${CKPT_DIR:-pre-trained/fpfn-heads}
RESULT_DIR=${RESULT_DIR:-fpfn_head}

cd "$(dirname "$0")"
for ITER in $ITERATIONS; do
    NAME="${NAME_OVERRIDE:-fpfn}_$ITER"
    for SPLIT in $SPLITS; do
        echo "[$(date +%H:%M)] FP/FN head: pmri/$SPLIT name=$NAME"
        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_fpfn_head.py \
            +train="$TRAIN" \
            +test="$TEST" \
            +data=pmri \
            ++data.split="$SPLIT" \
            ++data.non_empty_target=True \
            ++data.train_transforms=global_transforms \
            ++data.batch_size="$BATCH_SIZE" \
            +unet=monai_unet \
            ++unet.out_channels=1 \
            +model=fpfn_head \
            ++model.num_classes=2 \
            ++model.name="$NAME" \
            ++model.perturbations="$PERTURBATION" \
            ++model.adversarial_skip_epochs="$ADVERSARIAL_SKIP_EPOCHS" \
            ++model.adversarial_warmup_epochs="$ADVERSARIAL_WARMUP_EPOCHS" \
            ++model.adv_target_dice_drop="$ADV_TARGET_DICE_DROP" \
            ++model.count_target="$COUNT_TARGET" \
            ++model.adv_down="$ADV_DOWN" \
            ++model.adv_down_target_log_drop="$ADV_DOWN_TARGET_LOG_DROP" \
            ++model.regression_loss="$REGRESSION_LOSS" \
            +trainer=score_predictor_trainer \
            ++trainer.limit_train_batches="$N_BATCHES" \
            ++trainer.max_epochs="$N_EPOCHS" \
            ++trainer.ckpt_start_epoch="$CKPT_START_EPOCH" \
            ++trainer.logging="$LOGGING" \
            ++trainer.enable_train_loss_ckpt=False \
            ++trainer.model_checkpoint.dirpath="$CKPT_DIR" \
            ++trainer.result_dir="$RESULT_DIR"
    done
done
echo "[$(date +%H:%M)] FP/FN head training done"
