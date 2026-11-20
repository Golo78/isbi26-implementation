#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 9 -- PSEUDO-LABEL ORACLE: the exp8 oracle term
#
#     log(1 + FP + FN),   FP = sum p (1 - g_hat),   FN = sum (1 - p) g_hat
#
# but against a CORRUPTED label map g_hat instead of the GT (fixed per image, no gradient).
# This is the form a dense pseudo-label / error-map head would plug into: the gradient
# reaches the adapters only through the mask, exactly as for the oracle.  The sweep answers
# how good g_hat has to be before the TTA stops helping.
#
#   PSEUDO_MODE=recall  g_hat = P_0 with a fraction PSEUDO_LEVEL of its error pixels corrected
#                       (spatially coherent blobs); 1.0 = exp8 oracle, 0.0 = g_hat = P_0
#   PSEUDO_MODE=morph   g_hat = GT dilated (PSEUDO_LEVEL > 0) / eroded (< 0) by |level| px
#
# Same setup as exp8 (objective=none, joint form, post-norm, sites [0,17], 100 steps).
# Writes to results/latent_tta_pseudo_<mode><level>/<ckpt>/latent_tta.pkl.
#
#   usage: PSEUDO_MODE=recall PSEUDO_LEVEL=0.5 [CKPT_NAME=cleanHalf_0] [SPLITS=threet-to-onepointfivet] bash exp9_pseudo_oracle.sh

export OBJECTIVE=none
export PLACEMENT=post
export KAPPA_WEIGHT=0.0
export SIZE_WEIGHT=0.0
export FPFN_SOURCE=pseudo
export FPFN_WEIGHT=1.0
export FPFN_FORM=joint
export FPFN_FN_WEIGHT=1.0
export PSEUDO_MODE=${PSEUDO_MODE:-recall}
export PSEUDO_LEVEL=${PSEUDO_LEVEL:-1.0}
export CKPT_SUBDIR=${CKPT_SUBDIR:-pre_norm_with_logits}
export RESULT_DIR=${RESULT_DIR:-latent_tta_pseudo_${PSEUDO_MODE}${PSEUDO_LEVEL}}
export MAX_STEPS=${MAX_STEPS:-100}
exec bash "$(dirname "$0")/eval_latent_tta.sh"
