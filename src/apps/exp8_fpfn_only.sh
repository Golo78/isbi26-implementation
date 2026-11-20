#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 8 -- FP/FN only: no mu-hat (and no kappa) in the objective, only the descent on
#
#     log(1 + FP + a * FN)          (joint form, anchored at 0: minimal only for a perfect mask)
#
#   FPFN_SOURCE=oracle  true soft FP / FN of the current mask (labels; ceiling of the idea)
#   FPFN_SOURCE=head    hard-count FP/FN head fpfn_<k> (model/fpfn_head.py), paired with the
#                       score predictor cleanHalf_<k>; label-free
#
# With a single term the Adam step does not depend on its weight (up to eps), so there is no
# lambda to tune.  The score predictor is still loaded: mu-hat / kappa-hat are recorded per
# step for comparison with exp6 / exp7, but never enter the objective (objective=none).
# Same setup as exp6 / exp7 (pre-norm predictor, sites [0,17], post-norm affine, 100 steps).
# Writes to results/latent_tta_fpfnonly_<source>[_fn<a>]/<ckpt>/latent_tta.pkl.
#
#   usage: [FPFN_SOURCE=oracle] [FPFN_FN_WEIGHT=1.0] [CKPT_NAME=cleanHalf_0] [MAX_STEPS=100] bash exp8_fpfn_only.sh

export OBJECTIVE=none
export PLACEMENT=post
export KAPPA_WEIGHT=0.0
export SIZE_WEIGHT=0.0
export FPFN_SOURCE=${FPFN_SOURCE:-oracle}
export FPFN_WEIGHT=1.0
export FPFN_FORM=joint
export FPFN_FN_WEIGHT=${FPFN_FN_WEIGHT:-1.0}
export CKPT_SUBDIR=${CKPT_SUBDIR:-pre_norm_with_logits}
if [[ $FPFN_FN_WEIGHT == 1.0 || $FPFN_FN_WEIGHT == 1 ]]; then _fnw=""; else _fnw="_fn$FPFN_FN_WEIGHT"; fi
export RESULT_DIR=${RESULT_DIR:-latent_tta_fpfnonly_${FPFN_SOURCE}${_fnw}}
export MAX_STEPS=${MAX_STEPS:-100}
exec bash "$(dirname "$0")/eval_latent_tta.sh"
