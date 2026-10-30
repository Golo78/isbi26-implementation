#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 2 -- move the latent adapter from post-norm to pre-norm.
#
# The adapter is registered as a forward PRE hook on the swivel's InstanceNorm and
# prepended, so it runs before the score predictor's recording hook: the QC head sees the
# adapted features while the segmentation path sees IN(gamma*x + beta) == IN(x).
#
# What to read in the output:
#   `logit drift`  ~0  -> the affine was annihilated; any mu-hat gain is head-side only
#                         and cannot translate into a Dice gain.
#   `logit drift`  >0  -> the annihilation argument does not hold at these sites.
#
# Independent of the other two experiments: objective stays mu-hat, kappa term off.
# Writes to results/latent_tta_prenorm_adapter/<ckpt>/latent_tta.pkl (not latent_tta_prenorm,
# which holds the runs with the pre-norm *score predictor*, see run_tta100_queue.sh).
#
#   usage: [SPLITS=...] [CKPT_NAME=cleanHalf_0] [MAX_STEPS=100] bash exp2_prenorm_adapter.sh

export OBJECTIVE=mu
export PLACEMENT=pre
export KAPPA_WEIGHT=0.0
export RESULT_DIR=${RESULT_DIR:-latent_tta_prenorm_adapter}
export MAX_STEPS=${MAX_STEPS:-100}
exec bash "$(dirname "$0")/eval_latent_tta.sh"
