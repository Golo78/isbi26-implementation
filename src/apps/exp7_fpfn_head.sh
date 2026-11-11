#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 7 -- FP/FN head: exp6 with the true FP/FN replaced by a trained FP/FN head.
#
#     mu-hat - FPFN_WEIGHT * log(1 + FP_hat + a * FN_hat)     (FPFN_FORM=joint, default here)
#
# FP_hat / FN_hat come from the FP/FN head (model/fpfn_head.py, trained by
# train_fpfn_head.sh) that reads the same features as the score predictor; head fpfn_<k> is
# paired with score predictor cleanHalf_<k>.  Label-free, i.e. a deployable method; compare
# with the oracle of the same form (results/latent_tta_mu_fpfn_joint_w<w>).  The true soft
# FP / FN are still recorded for analysis.
# Writes to results/latent_tta_mu_fpfnhead[_<form>][_fn<a>]_w<FPFN_WEIGHT>/<ckpt>/latent_tta.pkl.
#
#   usage: [FPFN_WEIGHT=1.0] [FPFN_FORM=joint] [FPFN_FN_WEIGHT=1.0] [CKPT_NAME=cleanHalf_0] [MAX_STEPS=100] bash exp7_fpfn_head.sh

export OBJECTIVE=mu
export PLACEMENT=post
export KAPPA_WEIGHT=0.0
export SIZE_WEIGHT=0.0
export FPFN_SOURCE=head
export FPFN_WEIGHT=${FPFN_WEIGHT:-1.0}
export FPFN_FORM=${FPFN_FORM:-joint}
export FPFN_FN_WEIGHT=${FPFN_FN_WEIGHT:-1.0}
export CKPT_SUBDIR=${CKPT_SUBDIR:-pre_norm_with_logits}
if [[ $FPFN_FORM == joint ]]; then _form=""; else _form="_$FPFN_FORM"; fi
if [[ $FPFN_FN_WEIGHT == 1.0 || $FPFN_FN_WEIGHT == 1 ]]; then _fnw=""; else _fnw="_fn$FPFN_FN_WEIGHT"; fi
export RESULT_DIR=${RESULT_DIR:-latent_tta_mu_fpfnhead${_form}${_fnw}_w${FPFN_WEIGHT}}
export MAX_STEPS=${MAX_STEPS:-100}
exec bash "$(dirname "$0")/eval_latent_tta.sh"
