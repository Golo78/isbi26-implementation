#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 1 -- oracle upper bound.
#
# Replaces the expected Dice (mu-hat) in the gradient ascent by the REAL Dice (a
# differentiable soft Dice computed against the ground-truth labels).  Not applicable in
# practice, but it answers two questions:
#   * what could a perfect quality estimator achieve with this adaptation family?
#   * how much of the achieved improvement hinges on the mu-hat / true-Dice correlation
#     (= the gap between this run and the mu-hat baseline)?
#
# Independent of the other two experiments: placement stays post-norm, kappa term off.
# Writes to results/latent_tta_oracle/<ckpt>/latent_tta.pkl.
#
#   usage: [SPLITS=...] [CKPT_NAME=cleanHalf_0] [MAX_STEPS=100] bash exp1_oracle_dice.sh
#
# Run the mu-hat baseline with the SAME steps/sites for the comparison, e.g.
#   RESULT_DIR=latent_tta_baseline MAX_STEPS=$MAX_STEPS bash eval_latent_tta.sh

export OBJECTIVE=oracle-dice
export PLACEMENT=post
export KAPPA_WEIGHT=0.0
export RESULT_DIR=${RESULT_DIR:-latent_tta_oracle}
export MAX_STEPS=${MAX_STEPS:-100}
exec bash "$(dirname "$0")/eval_latent_tta.sh"
