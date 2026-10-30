#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 4 -- post-norm score predictor, affine transform also on the logits.
#
# Runs the mu-hat ascent with the post-norm checkpoints (pre-trained/score-predictors/
# post-norm_with_logits) twice over the same images, each into its own result dir:
#   latent_tta_postnorm         latent sites [0,17]            (reference)
#   latent_tta_postnorm_logits  latent sites [0,17] + logits
# Set ADAPT_SITES="[]" for a logits-only run (then also change RESULT_DIR).
#
# Every run additionally computes the threshold oracle and writes the per-example plots
# (predicted vs. actual Dice, kappa, masks) to results/<RESULT_DIR>/<ckpt>/plots/.
#
#   usage: [SPLITS=...] [CKPT_NAME=cleanHalf_0] [MAX_STEPS=200] [CUDA_DEVICE=3] bash exp4_logit_affine.sh

export OBJECTIVE=mu
export PLACEMENT=post
export KAPPA_WEIGHT=0.0
export CKPT_SUBDIR=${CKPT_SUBDIR:-post-norm_with_logits}
export MAX_STEPS=${MAX_STEPS:-200}
HERE="$(dirname "$0")"

if [ "${SKIP_REFERENCE:-False}" != "True" ]; then
    ADAPT_LOGITS=False RESULT_DIR=${RESULT_DIR_REF:-latent_tta_postnorm} bash "$HERE/eval_latent_tta.sh"
fi
ADAPT_LOGITS=True RESULT_DIR=${RESULT_DIR:-latent_tta_postnorm_logits} bash "$HERE/eval_latent_tta.sh"
