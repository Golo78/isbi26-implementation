#!/usr/bin/env bash
set -euo pipefail
# EXPERIMENT 5 -- size oracle: mu-hat ascent + size anchor from the true FP / FN counts.
#
# Before the adaptation, the true FP_0 and FN_0 of the un-adapted mask give the target
# size G_hat = |P_0| - FP_0 + FN_0 (= |G|).  The ascent on mu-hat is then combined with a
# simultaneous descent on the log-size error of the soft mask against G_hat:
#
#     mu-hat - SIZE_WEIGHT * (log(1 + sum p) - log(1 + G_hat))^2
#
# mu-hat decides WHERE the mask changes, the size term HOW MUCH.  Uses labels, so it is an
# oracle: it answers whether FP/FN information *in addition to* mu-hat would help if a
# head could predict it.  SIZE_NOISE > 0 multiplies FP_0 and FN_0 by independent
# (1 + U(-SIZE_NOISE, SIZE_NOISE)) factors: how accurate would such a head have to be?
#
# Same setup as the mu-hat baseline (results/latent_tta_prenorm): pre-norm predictor,
# sites [0,17], post-norm affine, kappa term off.
# Writes to results/latent_tta_mu_size_w<SIZE_WEIGHT>[_noise<SIZE_NOISE>]/<ckpt>/latent_tta.pkl.
#
#   usage: [SIZE_WEIGHT=1.0] [SIZE_NOISE=0.0] [CKPT_NAME=cleanHalf_0] [MAX_STEPS=100] bash exp5_size_oracle.sh

export OBJECTIVE=mu
export PLACEMENT=post
export KAPPA_WEIGHT=0.0
export SIZE_WEIGHT=${SIZE_WEIGHT:-1.0}
export SIZE_NOISE=${SIZE_NOISE:-0.0}
export CKPT_SUBDIR=${CKPT_SUBDIR:-pre_norm_with_logits}
NOISE_TAG=""
if [ "$SIZE_NOISE" != "0" ] && [ "$SIZE_NOISE" != "0.0" ]; then NOISE_TAG="_noise${SIZE_NOISE}"; fi
export RESULT_DIR=${RESULT_DIR:-latent_tta_mu_size_w${SIZE_WEIGHT}${NOISE_TAG}}
export MAX_STEPS=${MAX_STEPS:-100}
exec bash "$(dirname "$0")/eval_latent_tta.sh"
