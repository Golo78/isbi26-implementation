#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 6 -- FP/FN oracle: mu-hat ascent + descent on the true FP and FN at every step.
#
# At every ascent step the soft FP and FN of the CURRENT mask are computed from the labels
# (FP = sum p(1-g), FN = sum (1-p)g) and pushed towards 0 together with the ascent on mu-hat:
#
#     FPFN_FORM=separate (default):  mu-hat - FPFN_WEIGHT * (log(1 + FP) + log(1 + FN))
#     FPFN_FORM=joint:               mu-hat - FPFN_WEIGHT * log(1 + FP + FN)
#     FPFN_FN_WEIGHT=a weights FN inside the term, e.g. log(1 + FP + a * FN)
#
# The separate form is a log-product, minimal as soon as FP = 0 OR FN = 0: in practice the
# adaptation removes (almost) all FP and lets FN grow.  The joint form has no such shortcut.
#
# Simulates a head that predicts FP and FN perfectly for every mask the adaptation
# produces.  Unlike the size oracle (exp5, target fixed once before the adaptation) the
# gradient of FP / FN w.r.t. the mask is the label map, i.e. it carries *where* to change.
# Uses labels, so it is an upper bound, not a method.
#
# Same setup as exp5 (pre-norm predictor, sites [0,17], post-norm affine, kappa term off).
# Writes to results/latent_tta_mu_fpfn[_joint][_fn<a>]_w<FPFN_WEIGHT>/<ckpt>/latent_tta.pkl.
#
#   usage: [FPFN_WEIGHT=1.0] [FPFN_FORM=separate] [FPFN_FN_WEIGHT=1.0] [CKPT_NAME=cleanHalf_0] [MAX_STEPS=100] bash exp6_fpfn_oracle.sh

export OBJECTIVE=mu
export PLACEMENT=post
export KAPPA_WEIGHT=0.0
export SIZE_WEIGHT=0.0
export FPFN_WEIGHT=${FPFN_WEIGHT:-1.0}
export FPFN_FORM=${FPFN_FORM:-separate}
export FPFN_FN_WEIGHT=${FPFN_FN_WEIGHT:-1.0}
export CKPT_SUBDIR=${CKPT_SUBDIR:-pre_norm_with_logits}
if [[ $FPFN_FORM == separate ]]; then _form=""; else _form="_$FPFN_FORM"; fi
if [[ $FPFN_FN_WEIGHT == 1.0 || $FPFN_FN_WEIGHT == 1 ]]; then _fnw=""; else _fnw="_fn$FPFN_FN_WEIGHT"; fi
export RESULT_DIR=${RESULT_DIR:-latent_tta_mu_fpfn${_form}${_fnw}_w${FPFN_WEIGHT}}
export MAX_STEPS=${MAX_STEPS:-100}
exec bash "$(dirname "$0")/eval_latent_tta.sh"
