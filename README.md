# Quality Control for Medical Image Segmentation Under Domain Shift With Heteroscedastic Regression

Quality control (QC) experiments for segmentation reliability. This repo trains UNet models, trains score predictors for QC, and evaluates multiple QC baselines (score-agreement, Mahalanobis, and predictive-entropy) with a shared analysis notebook.

## Project layout
- models
	- [src/model/unet](src/model/unet) - U-Net models.
	- [src/model/score_predictor](src/model/score_adapter) - score prediction implementation.
	- [src/model/mahalanobis](src/model/mahalanobis) - Mahalanobis distance model.
	- [src/model/calibration](src/model/calibration) - calibration utilities.
- [src/apps](src/apps) - training and evaluation entrypoints.
- [src/notebooks](src/notebooks) - evaluation notebooks; [src/notebooks/results](src/notebooks/results) - result analyses of the TTA experiments.
- [results/](results/) - saved outputs (per-dataset/split/method runs).
- [pre-trained/](pre-trained/) - pretrained checkpoints.

## Installation
TODO

## Training 
UNet training and eval:
- [src/apps/train_unet_per_split.sh](src/apps/train_unet_per_split.sh)

Score predictor training and eval (Beta$_{\mu,\kappa}$ QC head on top of UNet):
- [src/apps/train_score_predictor.sh](src/apps/train_score_predictor.sh)

FP/FN head training and eval ([src/model/fpfn_head.py](src/model/fpfn_head.py)): predicts
log(1 + FP) and log(1 + FN) of the U-Net's mask (`COUNT_TARGET=hard`: thresholded mask,
default; `soft`: probabilities as in the FP/FN oracle), MSE in log space, with the same
architecture, swivels and hook placement as the score predictor (head `fpfn_<k>` is paired
with `cleanHalf_<k>`), trained separately from it, and adversarial training analogous to the
score predictor's location adversary (input steps that raise the predicted error; the step
size is controlled so that the true hard Dice drops by 0.1, as for the score predictor):
- [src/apps/train_fpfn_head.sh](src/apps/train_fpfn_head.sh) (one GPU) /
  [src/apps/run_fpfn_head_queue.sh](src/apps/run_fpfn_head_queue.sh) (one head index per GPU)
- checkpoints: `pre-trained/fpfn-heads/`, evaluation (MAE / correlation in log space, soft
  Dice implied by the predicted counts): `results/fpfn_head/<ckpt>/eval_data.pkl`
- `ADV_DOWN=True`: second adversary in the TTA direction (input steps that LOWER the
  predicted error, trained on with their true counts), so that the head does not give way
  where the TTA would exploit it; its step size is controlled so that the predicted
  log(1 + FP + FN) drops by `ADV_DOWN_TARGET_LOG_DROP` (0.5) per step
- batch per step: 30 clean + 30 adversarial images (+ 30 with `ADV_DOWN`). Unlike the score
  predictor (30 clean + 15 location + 15 concentration), ONE adversary along the joint error
  log(1 + FP + FN) gets all 30; there is no separate FP / FN adversary (see the docstring of
  `fpfn_head.py` for why this may under-train FN directions)

## Evaluation baselines
These scripts compute QC signals and save them into results files for later aggregation:
- Score agreement (SA): [src/apps/eval_score_agreement.sh](src/apps/eval_score_agreement.sh)
- Mahalanobis distance (Maha): [src/apps/eval_mahalanobis.sh](src/apps/eval_mahalanobis.sh)
- Predictive entropy (PE): [src/apps/eval_comp_entropy.sh](src/apps/eval_comp_entropy.sh)

## Latent test-time adaptation (TTA)
Gradient ascent on the predicted score mu-hat of the QC head over a per-channel affine
gamma * z + beta of latent U-Net features (swivels `[0, 17]` by default), per image.
Generic entry point: [src/apps/eval_latent_tta.sh](src/apps/eval_latent_tta.sh); every
switch is an environment variable (see the header of the script), the defaults are in
[src/configs/tta/latent_affine.yaml](src/configs/tta/latent_affine.yaml). Every run writes
`results/<RESULT_DIR>/<ckpt>/latent_tta.pkl` plus per-example plots
([src/apps/plot_latent_tta.py](src/apps/plot_latent_tta.py)).

Two different "placements", not to be confused:
- **`model.hook_placement`** — where the *score predictor* reads the swivel features: `pre`
  (input of the swivel InstanceNorm, default) or `post` (its output). Fixed at training time
  (`HOOK_PLACEMENT` in [train_score_predictor.sh](src/apps/train_score_predictor.sh)) and
  stored in the checkpoint; the eval scripts always take it from there. Checkpoints live in
  `pre-trained/score-predictors/pre_norm_with_logits/` (default) and
  `pre-trained/score-predictors/post-norm_with_logits/`.
- **`tta.placement`** — where the *TTA affine* sits relative to the InstanceNorm: `post`
  (default) or `pre` (experiment 2 only).

### Experiments
Thin wrappers around `eval_latent_tta.sh` that fix the switches and the result directory:

1. **Oracle upper bound** — [src/apps/exp1_oracle_dice.sh](src/apps/exp1_oracle_dice.sh)
   (`++tta.objective=oracle-dice`). Replaces the expected Dice in the ascent by the *real*
   Dice (a differentiable soft Dice against the ground truth — the hard Dice has no
   gradient). Uses labels, so it is not deployable; it bounds what latent adaptation could
   achieve with a perfect quality estimate, and the gap to the mu-hat baseline is the price
   of the imperfect expected/real Dice correlation.
2. **Pre-norm adapter** — [src/apps/exp2_prenorm_adapter.sh](src/apps/exp2_prenorm_adapter.sh)
   (`++tta.placement=pre`). Moves the affine from the output of the swivel InstanceNorm to
   its input (prepended before the score predictor's recording hook, so the QC head still
   sees the adapted features). The reported `logit drift` says whether the segmentation
   moved at all.
3. **Kappa** — [src/apps/exp3_kappa_bounds.sh](src/apps/exp3_kappa_bounds.sh) first: the
   Beta credible bounds of the *current* predictions (empirical coverage, width, PIT,
   mu-hat/Dice correlation), no adaptation. Then
   [src/apps/exp3_kappa_descent.sh](src/apps/exp3_kappa_descent.sh)
   (`++tta.kappa_weight=...`), which descends the predictive uncertainty *simultaneously*
   with the ascent on the score, in one combined objective per step.
4. **Logit affine** — [src/apps/exp4_logit_affine.sh](src/apps/exp4_logit_affine.sh)
   (`++tta.adapt_logits=true`). Adds a per-channel affine on the U-Net logits; for the
   binary PMRI network this is a per-image threshold shift, compared against the threshold
   oracle that every run computes.

5. **Size oracle** — [src/apps/exp5_size_oracle.sh](src/apps/exp5_size_oracle.sh)
   (`++tta.size_weight=...`, `++tta.size_noise=...`). Ascent on mu-hat plus a simultaneous
   descent on the log-size error of the soft mask against G_hat = |P_0| − FP_0 + FN_0, taken
   once per image from the *true* FP / FN counts of the un-adapted mask. Uses labels, so it
   is not deployable; it tests whether FP/FN information *in addition to* mu-hat would
   help. `size_noise` multiplies FP_0 and FN_0 by (1 + U(−n, n)): how accurate a FP/FN head
   would have to be.

6. **FP/FN oracle** — [src/apps/exp6_fpfn_oracle.sh](src/apps/exp6_fpfn_oracle.sh)
   (`++tta.fpfn_weight=...`). Ascent on mu-hat plus a simultaneous descent on
   log(1 + FP) + log(1 + FN) of the *current* soft mask, recomputed from the labels at every
   step (anchor 0). Simulates a head that predicts FP and FN perfectly for every adapted
   mask; compared with the size oracle it separates knowing FP/FN once from knowing them
   at every step (the gradient of FP / FN carries *where* to change). This *separate* form
   is a log-product and already minimal when FP = 0 or FN = 0; in practice it removes almost
   all FP and lets FN grow (worse than mu-hat alone). `FPFN_FORM=joint`
   (`++tta.fpfn_form=joint`) descends log(1 + FP + FN) instead. `FPFN_FN_WEIGHT=a`
   (`++tta.fpfn_fn_weight=a`) weights FN inside the term, log(1 + FP + a * FN).

7. **FP/FN head** — [src/apps/exp7_fpfn_head.sh](src/apps/exp7_fpfn_head.sh)
   (`++tta.fpfn_source=head`). Experiment 6 with the true FP / FN replaced by the trained
   FP/FN head (joint form by default): label-free. The head is attached to the score
   predictor's wrapper and reads the features of the same forward pass, so mu-hat is
   unchanged and the latent adapters act on both heads. The true soft FP / FN are still
   recorded (`fp_soft_per_step`, `fn_soft_per_step`, hard: `fp_hard_per_step`,
   `fn_hard_per_step`) next to the predictions
   (`fp_pred_per_step`, `fn_pred_per_step`). Naming a head (`FPFN_HEAD_NAME`) with the
   default source `oracle` only records its predictions.

Every TTA run prints and stores the credible bounds before and after adaptation, so the
kappa view is available in all experiments, as well as the hard FP / FN counts and mask
sizes before and after adaptation (over- vs. under-segmentation).

### Batch runs
The 100-step comparison over all PMRI splits and predictors `cleanHalf_0..4`:
- [src/apps/run_tta100_queue.sh](src/apps/run_tta100_queue.sh) — 2x2 design
  (pre-/post-norm score predictor x with/without logit affine), one GPU, sequential.
- [src/apps/run_oracle_queue.sh](src/apps/run_oracle_queue.sh) — oracle runs (sites, sites +
  logits), distributed round-robin over several GPUs.
- [src/apps/run_size_oracle_queue.sh](src/apps/run_size_oracle_queue.sh) — size oracle
  (weights x noise levels x predictors), same round-robin scheme.
- [src/apps/run_fpfn_oracle_queue.sh](src/apps/run_fpfn_oracle_queue.sh) — FP/FN oracle
  (weights x predictors), same round-robin scheme.

Logs go to `logs/`.

### Result directories
| `results/...` | score predictor | adapted | objective | script |
|---|---|---|---|---|
| `latent_tta_prenorm` | pre-norm | sites | mu-hat | `run_tta100_queue.sh` |
| `latent_tta_prenorm_logits` | pre-norm | sites + logits | mu-hat | `run_tta100_queue.sh` |
| `latent_tta_postnorm` | post-norm | sites | mu-hat | `run_tta100_queue.sh`, `exp4_logit_affine.sh` |
| `latent_tta_postnorm_logits` | post-norm | sites + logits | mu-hat | `run_tta100_queue.sh`, `exp4_logit_affine.sh` |
| `latent_tta_oracle` | pre-norm | sites | true soft Dice | `run_oracle_queue.sh`, `exp1_oracle_dice.sh` |
| `latent_tta_oracle_logits` | pre-norm | sites + logits | true soft Dice | `run_oracle_queue.sh` |
| `latent_tta_prenorm_adapter` | pre-norm | sites, affine pre-norm | mu-hat | `exp2_prenorm_adapter.sh` |
| `kappa_bounds` | pre-norm | — | — | `exp3_kappa_bounds.sh` |
| `latent_tta_kappa_<obj>_w<weight>` | pre-norm | sites | mu-hat + uncertainty | `exp3_kappa_descent.sh` |
| `latent_tta_mu_size_w<weight>[_noise<n>]` | pre-norm | sites | mu-hat + size from true FP/FN | `exp5_size_oracle.sh`, `run_size_oracle_queue.sh` |
| `latent_tta_mu_fpfn_w<weight>` | pre-norm | sites | mu-hat + true log(1+FP) + log(1+FN) at every step | `exp6_fpfn_oracle.sh`, `run_fpfn_oracle_queue.sh` |
| `latent_tta_mu_fpfn_joint_w<weight>` | pre-norm | sites | mu-hat + true log(1+FP+FN) at every step | `FPFN_FORM=joint` with the same scripts |
| `latent_tta_mu_fpfn_joint_fn<a>_w<weight>` | pre-norm | sites | mu-hat + true log(1+FP+a*FN) at every step | `FPFN_FORM=joint FPFN_FN_WEIGHT=a` |
| `latent_tta_mu_fpfnhead[_<form>][_fn<a>]_w<weight>` | pre-norm | sites | mu-hat + predicted log(1+FP+a*FN) (FP/FN head) | `exp7_fpfn_head.sh` |
| `fpfn_head` | — | — | FP/FN head evaluation (no TTA) | `train_fpfn_head.sh` |

## Analysis notebooks
QC analysis workflow is in [src/notebooks/QC_eval.ipynb](src/notebooks/QC_eval.ipynb). It:
- Loads results for multiple datasets/splits and all runs.
- Fits calibrators (thresholding for correlation-based methods; beta adapters for beta-based predictors).
- Computes ranking metrics (Pearson’s $\rho$, MAE, eAURC) and risk-control metrics (Rec<sup>+</sup> / Rec<sup>-</sup> at t=0.8, α=0.95).

UNet evaluation is in [src/notebooks/unet_eval.ipynb](src/notebooks/unet_eval.ipynb).

Latent TTA:
- [src/notebooks/latent_tta_experiments.ipynb](src/notebooks/latent_tta_experiments.ipynb) — overview of the TTA runs and experiments.
- [src/notebooks/results/latent_tta/](src/notebooks/results/latent_tta/) — per-case analysis, harm and stopping rules, segmentation maps.
- [src/notebooks/results/latent_tta_20_steps/](src/notebooks/results/latent_tta_20_steps/) — early comparison of 20 / 100 / 1000 ascent steps.
- [src/notebooks/results/meetings/](src/notebooks/results/meetings/) — status notebooks for the meetings (in German).

The notebooks resolve paths relative to their own location and read from [results/](results/).

## Notes
- Results are organized by dataset, split, method, and run ID under [results/](results/).
- Dataset shifts used in the paper:
	- M\&Ms scanner drift → `scanner-symphonytim`
	- M\&Ms pathology drift → `pathology-norm-vs-fall-scanners-all`
	- PMRI dataset shift → `promise12`
	- PMRI 3T→1.5T shift → `threet-to-onepointfivet`