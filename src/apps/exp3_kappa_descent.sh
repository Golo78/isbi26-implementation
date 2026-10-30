#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 3b -- take kappa into the loop.
#
# Descends the predictive uncertainty of the QC head SIMULTANEOUSLY with the ascent on the
# predicted Dice, in one combined objective per step:
#
#   KAPPA_OBJECTIVE=sigma      (default)  obj = mu - w * sqrt(mu(1-mu)/(kappa+1))
#   KAPPA_OBJECTIVE=log-kappa             obj = mu + w * log(kappa)
#
# Look at the kappa credible bounds first (exp3_kappa_bounds.sh) -- if the head is already
# overconfident (empirical coverage far below nominal), pushing kappa up further is
# expected to make the bounds worse, and that is exactly what the before/after coverage
# printed by this run measures.
#
# Independent of the other two experiments: objective stays mu-hat, placement post-norm.
# Writes to results/latent_tta_kappa_w<weight>/<ckpt>/latent_tta.pkl.
#
#   usage: [KAPPA_WEIGHT=0.1] [KAPPA_OBJECTIVE=sigma] [MAX_STEPS=100] bash exp3_kappa_descent.sh

export OBJECTIVE=mu
export PLACEMENT=post
export KAPPA_WEIGHT=${KAPPA_WEIGHT:-0.1}
export KAPPA_OBJECTIVE=${KAPPA_OBJECTIVE:-sigma}
export MAX_STEPS=${MAX_STEPS:-100}
export RESULT_DIR=${RESULT_DIR:-latent_tta_kappa_${KAPPA_OBJECTIVE}_w${KAPPA_WEIGHT}}
exec bash "$(dirname "$0")/eval_latent_tta.sh"
