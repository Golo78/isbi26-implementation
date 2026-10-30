#!/usr/bin/env bash
set -euo pipefail

# Behavior
CUDA_DEVICE=${CUDA_DEVICE:-7}
TRAIN=${TRAIN:-True}
TEST=${TEST:-True}
LOGGING=${LOGGING:-True}
TRAIN_LOSS_CKPT=${TRAIN_LOSS_CKPT:-False}  # set to False for validation-only checkpointing during debugging

# Training 
NON_EMPTY_TARGET=True
N_BATCHES=${N_BATCHES:-100}
N_EPOCHS=${N_EPOCHS:-400}
WIDEN_FACTOR=1
LAYERS=8
TRAIN_TRANSFORMS='global_transforms'
DIST='all'

DATASETS=${DATASETS:-"pmri"}
LOCATION_TARGET_SCORE_DELTA=${LOCATION_TARGET_SCORE_DELTA:-0.1}
# Likelihood switch for gaussian vs beta regression
LIKELIHOOD=${LIKELIHOOD:-beta}       # beta | betahomoscedastic | gaussian | mse
LIKELIHOODS=${LIKELIHOODS:-$LIKELIHOOD}
PERTURBATION='predictor'  # none | predictor (model and both removed)
PREDICTOR_PARAMETER=${PREDICTOR_PARAMETER:-location+concentration}  # location | concentration | location+concentration
CONCENTRATION_TARGET_LOG_DELTA=${CONCENTRATION_TARGET_LOG_DELTA:-0.6931471805599453}
CONCENTRATION_CONTROLLER_GAIN=${CONCENTRATION_CONTROLLER_GAIN:-0.1}
MIN_CONCENTRATION=${MIN_CONCENTRATION:-1e-6}

# Beta decay and adversarial warmup settings - conditionally set based on perturbation type
# For non-adversarial: skip and warmup are 0, only decay matters
# For adversarial: all three phases are used
if [[ "$PERTURBATION" == "none" ]]; then
    ADVERSARIAL_SKIP_EPOCHS=${ADVERSARIAL_SKIP_EPOCHS:-0}
    ADVERSARIAL_WARMUP_EPOCHS=${ADVERSARIAL_WARMUP_EPOCHS:-0}
    BETA_DECAY_EPOCHS=${BETA_DECAY_EPOCHS:-50} # Decay over 50 epochs for non-adversarial
    BETA_FINAL=${BETA_FINAL:-0.0}
else
    ADVERSARIAL_SKIP_EPOCHS=${ADVERSARIAL_SKIP_EPOCHS:-40}     # epochs with no adversarial perturbations
    ADVERSARIAL_WARMUP_EPOCHS=${ADVERSARIAL_WARMUP_EPOCHS:-30} # linear ramp epochs after skip (0 = no ramp)
    BETA_DECAY_EPOCHS=${BETA_DECAY_EPOCHS:-20}                 # linear decay of beta after warmup (0 = jump to BETA_FINAL immediately)
    BETA_FINAL=${BETA_FINAL:-0.0}                              # final beta value after decay (beta=1 during skip+warmup)
fi


PREDICTION_HEAD=True
HIDDEN_DIM=64
SWIVELS='first-embedding-last'
HOOK_PLACEMENT=${HOOK_PLACEMENT:-pre}  # pre = swivel input (pre-InstanceNorm), post = swivel output (post-norm)
CKPT_NAME=''

ITERATIONS='0 1 2 3 4'

for ITER in ${ITERATIONS:-0}; do
    echo "Running iteration: $ITER"

    # for LOSS_FN in 'surface'; do
    for LOSS_FN in 'dice'; do
    # for LOSS_FN in 'dice'; 
        echo "Running experiments for loss function: $LOSS_FN"

        for DATA in ${DATASETS}; do
            echo "Running experiments for dataset: $DATA"

            # Defaults per dataset
            if [ "$DATA" = 'mnmv2' ]; then
                UNET_OUT_DIM=4; PRED_OUT_DIM=3; NUM_CLASSES=4
                DEFAULT_SPLITS=(
                    "scanner-symphonytim"
                    "pathology-norm-vs-fall-scanners-all"
                    # "phase-exclusive-scanners-all"
                )
            elif [ "$DATA" = 'pmri' ]; then
                UNET_OUT_DIM=1; PRED_OUT_DIM=3; NUM_CLASSES=2
                DEFAULT_SPLITS=(
                    "promise12"
                #   "promise12-i2cvb"
                    "threet-to-onepointfivet"
                    "onepointfivet-to-threet"
                )
            else
                echo "Unknown dataset '$DATA'" >&2; exit 2
            fi

            # Resolve splits list for this dataset
            if [ -n "${SPLITS:-}" ]; then
                # shellcheck disable=SC2206
                TARGET_SPLITS=(${SPLITS})
            else
                TARGET_SPLITS=("${DEFAULT_SPLITS[@]}")
            fi

            NAME="${NAME_OVERRIDE:-cleanHalf}_$ITER"

            if [[ "$PERTURBATION" == "predictor" ]]; then
                BATCH_SIZE=30
            else
                BATCH_SIZE=60
            fi
            for LIK in ${LIKELIHOODS}; do
                echo "Running experiments for likelihood: $LIK"
                for SPLIT in "${TARGET_SPLITS[@]}"; do
                    echo "Training score predictor for dataset=$DATA split=$SPLIT likelihood=$LIK"
                    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_score_predictor.py \
                    +train="$TRAIN" \
                    +test="$TEST" \
                    +data="$DATA" \
                    ++data.split="$SPLIT" \
                    ++data.non_empty_target="$NON_EMPTY_TARGET" \
                    ++data.train_transforms="$TRAIN_TRANSFORMS" \
                    ++data.batch_size="$BATCH_SIZE" \
                    +unet=monai_unet \
                    ++unet.out_channels="$UNET_OUT_DIM" \
                    +model=score_predictor \
                    ++model.adapter_output_dim="$HIDDEN_DIM" \
                    ++model.num_classes="$NUM_CLASSES" \
                    ++model.prediction_head_output_dim="$PRED_OUT_DIM" \
                    ++model.name="$NAME" \
                    ++model.adversarial_skip_epochs="$ADVERSARIAL_SKIP_EPOCHS" \
                    ++model.adversarial_warmup_epochs="$ADVERSARIAL_WARMUP_EPOCHS" \
                    ++model.beta_decay_epochs="$BETA_DECAY_EPOCHS" \
                    ++model.beta_final="$BETA_FINAL" \
                    ++model.location_target_score_delta="$LOCATION_TARGET_SCORE_DELTA" \
                    ++model.perturbations="$PERTURBATION" \
                    ++model.predictor_parameter="$PREDICTOR_PARAMETER" \
                    ++model.concentration_target_log_delta="$CONCENTRATION_TARGET_LOG_DELTA" \
                    ++model.concentration_controller_gain="$CONCENTRATION_CONTROLLER_GAIN" \
                    ++model.min_concentration="$MIN_CONCENTRATION" \
                    ++model.likelihood="$LIK" \
                    ++model.loss_fn="$LOSS_FN" \
                    ++model.widen_factor="$WIDEN_FACTOR" \
                    ++model.layers="$LAYERS" \
                    ++model.swivels="$SWIVELS" \
                    ++model.hook_placement="$HOOK_PLACEMENT" \
                    +trainer=score_predictor_trainer \
                    ++trainer.limit_train_batches="$N_BATCHES" \
                    ++trainer.max_epochs="$N_EPOCHS" \
                    ++trainer.logging="$LOGGING" \
                    ++trainer.enable_train_loss_ckpt="$TRAIN_LOSS_CKPT" \
                    ++trainer.result_dir="new_adversarial_strategy" \
                    ++trainer.ckpt_name="$CKPT_NAME"
            done
            done
            echo "Completed Dataset $DATA"
        done
        echo "Completed loss function $LOSS_FN"
    done
    echo "Completed iteration $ITER"
done
echo "All experiments completed!"
