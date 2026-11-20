"""FP/FN head: predicts the false-positive and false-negative counts of the U-Net mask.

Label-free counterpart of the FP/FN oracle of the latent TTA (`run_latent_tta`,
`fpfn_weight > 0`): there the TRUE counts of the current mask enter the objective at every
step.  This head estimates them per foreground class from the U-Net features, so that the
oracle can be replaced by a deployable estimate.  `count_target` selects what is counted:

    'hard' (default) : pixels of the thresholded mask (logits > 0 / argmax), like the hard
                       Dice the score predictor is trained on
    'soft'           : FP = sum p (1 - g), FN = sum (1 - p) g with the probabilities p --
                       exactly the quantities of the FP/FN oracle

Architecture
------------
Identical to the score (mu-hat) predictor -- same swivels, same hook placement, same
feature fuser + ResNet backbone + MLP (`PredictionHead`) -- only the output differs:
per foreground class (log(1 + FP_hat), log(1 + FN_hat)), kept >= 0 by a softplus.  It is
trained separately (the score predictor stays untouched), and at test time it is attached
to the score predictor's wrapper as an extra head (`attach_fpfn_head`): both heads then
read the features of the same forward pass, so the latent adapters act on both.

Training
--------
Regression in log space (MSE or Huber on log(1 + count)), targets computed from the
frozen U-Net's own mask.  Optionally adversarial, analogous to the location adversary
of the score predictor (`perturbations='predictor'`): every step, the clean images are
moved in input space along the gradient that RAISES the predicted error
log(1 + FP_hat + a * FN_hat) (the score predictor steps along the gradient that lowers
mu-hat), and the moved images are trained on in the next step with their true counts.
The step size is set by the same controller as the score predictor's: the TRUE hard Dice
of the moved images has to drop by `adv_target_dice_drop` (0.1) on average.  The head
thereby sees masks along exactly the directions in which its own prediction is most
sensitive.

Batch per step (batch size B = 30): B clean images + B moved images (ALL clean images of the
previous step, one adversary) [+ B from the down adversary below] = 2B [3B].  The score
predictor (`location+concentration`) also trains on 2B, but splits the clean images:
B/2 location (mu-hat down) + B/2 concentration (kappa).  Same adversarial share, but here a
single adversary along the JOINT error gets all B images; there is no separate FP and FN
adversary.  The joint gradient is dominated by the larger predicted count
(d log(1 + FP_hat + FN_hat) = (dFP_hat + dFN_hat) / (1 + FP_hat + FN_hat); FP_hat 797 vs.
FN_hat 320 on 3T->1.5T), so directions in which FN rises are rarely trained on -- a
plausible, unverified cause of the ~5x FN underestimation under shift.

`adv_mode='split'` replaces the joint adversary by one FP and one FN adversary, split like
the score predictor's location / concentration adversaries: the first B/2 clean images move
along +d log(1 + FP_hat) / dx, the other B/2 along +d log(1 + FN_hat) / dx, each half with its
own step-size controller on the same target (true hard Dice drop of `adv_target_dice_drop`).
Same number of adversarial images (B) as the joint mode.  Both modes log, per adversary,
how much the TRUE log(1 + FP) and log(1 + FN) of the moved images rose
(`adv[_fp|_fn]_true_dlog_fp|fn`), i.e. whether the FN direction is actually explored.

Optional second adversary in the TTA direction (`adv_down=True`): the clean images are also
moved along the gradient that LOWERS the predicted error -- what the latent TTA does at test
time -- and trained on with their true counts.  Where the head would report an improvement
that the U-Net's mask does not show, the regression loss is large, so the head learns not to
give way in the directions the TTA exploits.  The true Dice cannot steer this step (ideally
it does not change), so its size is controlled on the PREDICTED error, analogous to the
score predictor's concentration adversary (predicted log kappa drops by ln 2): the predicted
log(1 + FP_hat + a * FN_hat) of the moved images has to be `adv_down_target_log_drop` (0.5)
below that of their clean originals.  0.5 in log error is roughly the size of the upward
step (a true Dice drop of 0.1 at Dice ~0.86 multiplies FP + FN by ~1.7, log ~0.54) and about
what 100 TTA steps take off the head's prediction (-0.63 to -0.67 on 3T->1.5T).

Separate FP and FN heads (`count_output='fp'` / `'fn'`): the head is trained on ONE count only
-- the regression loss covers only that output, and the (joint-mode) adversary moves the clean
images along +d log(1 + FP_hat) / dx resp. +d log(1 + FN_hat) / dx; with `adv_down`, the down
adversary lowers that count.  The network keeps its two outputs (same architecture and
loading code); the other output receives no gradient and is meaningless.  For the TTA, an FP
head and an FN head are combined into one (`CombinedFPFNHead`, `attach_fpfn_head(...,
fn_module=...)`), which returns (log(1+FP_hat) of the FP head, log(1+FN_hat) of the FN head).
`adv_mode='split'` is not available for single-count heads.

Learning-rate schedule: `ReduceLROnPlateau` monitors `lr_monitor`.  The default 'val_loss'
includes the adversarial loss from epoch adversarial_skip_epochs + 1 on; that jump is never
undone, so the scheduler cuts the LR 4x by 10 within ~100 epochs and the head stops learning
(3.16e-5 -> 3.16e-9 in all heads trained so far).  'val_loss_clean' monitors the clean part only.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional

import lightning as L
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append('../')
from utils import find_shapes_for_swivels
from losses import soft_fp_fn_from_logits, soft_size_from_logits, fp_fn_counts
from model.score_adapter import (
    PredictionHead,
    ScorePredictionAdapter,
    ScorePredictionWrapper,
    resolve_score_swivels,
)


FPFN_HEAD = 'fpfn'  # key in ScorePredictionWrapper.extra_heads / extra_predictions


# --------------------------------------------------------------------------------------
# head + target helpers
# --------------------------------------------------------------------------------------

class FPFNPredictionHead(PredictionHead):
    """`PredictionHead` with a count output: [B, C-1, 2] = (log(1+FP_hat), log(1+FN_hat))."""

    def __init__(self, output_shapes, num_foreground: int, hidden_dim: int = 128,
                 widen_factor: int = 1, layers: int = 8):
        super().__init__(
            output_shapes=output_shapes,
            hidden_dim=hidden_dim,
            output_dim=2 * num_foreground,
            widen_factor=widen_factor,
            layers=layers,
        )
        self.num_foreground = int(num_foreground)
        # start near typical counts (log(1 + count) ~ 5, i.e. ~150 pixels) instead of
        # softplus(0) ~ 0.7, so the first epochs are not spent on the offset alone
        nn.init.constant_(self.fc.fc2.bias, 5.0)

    def forward(self, feats: dict) -> Tensor:
        x = self.backbone(self.feature_fuser(feats))
        out = self.fc(x)
        return F.softplus(out).view(out.size(0), self.num_foreground, 2)


COUNT_TARGETS = ('hard', 'soft')
ADV_MODES = ('joint', 'split')
COUNT_OUTPUTS = ('both', 'fp', 'fn')   # which count(s) the head is trained on
LR_MONITORS = ('val_loss', 'val_loss_clean')
_CHANNEL = {'fp': 0, 'fn': 1}
ADV_KIND = {'joint': 0, 'fp': 1, 'fn': 2}  # per-sample tag of the up adversary in eval_step


def hard_mask(logits: Tensor) -> Tensor:
    """[B, 1, H, W] labels: logits > 0 (binary) or argmax."""
    return (logits > 0).long() if logits.shape[1] == 1 else logits.argmax(1, keepdim=True)


@torch.no_grad()
def fpfn_targets(logits: Tensor, target: Tensor, num_classes: int, count_target: str = 'hard'):
    """True counts of the U-Net mask, no graph.

    Returns (log_counts [B, C-1, 2] = (log(1+FP), log(1+FN)), size [B, C-1] = |P|), counted
    on the hard mask or on the probabilities (`count_target`).
    """
    logits = logits.detach().float()
    if count_target == 'soft':
        fp, fn = soft_fp_fn_from_logits(logits, target, num_classes)
        size = soft_size_from_logits(logits)
    elif count_target == 'hard':
        fp, fn, size, _ = fp_fn_counts(hard_mask(logits), target, num_classes)
    else:
        raise ValueError(f'count_target must be one of {COUNT_TARGETS}, got {count_target!r}')
    return torch.stack([fp, fn], dim=-1).log1p(), size


@torch.no_grad()
def hard_dice(logits: Tensor, target: Tensor, num_classes: int) -> Tensor:
    """True hard Dice per sample [B], mean over foreground classes (empty vs. empty = 1)."""
    fp, fn, p, g = fp_fn_counts(hard_mask(logits.detach()), target, num_classes)
    tp = p - fp
    denom = p + g
    d = torch.where(denom > 0, 2 * tp / denom.clamp_min(1), torch.ones_like(denom))
    return d.mean(1)


def log_counts_to_counts(log_counts: Tensor):
    """(fp, fn) [B, C-1] from a [B, C-1, 2] log(1 + count) tensor (prediction or target)."""
    counts = torch.expm1(log_counts.float().clamp_min(0))
    return counts[..., 0], counts[..., 1]


def log_error(log_counts: Tensor, fn_weight: float = 1.0) -> Tensor:
    """log(1 + FP + a * FN) per sample [B], mean over foreground classes.

    The joint FP/FN term of the latent TTA, applied to log counts.
    """
    fp, fn = log_counts_to_counts(log_counts)
    return torch.log1p(fp + fn_weight * fn).mean(1)


def dice_from_counts(size: Tensor, fp: Tensor, fn: Tensor) -> Tensor:
    """Dice 2 TP / (|P| + |G|) from |P| = size, TP = |P| - FP, |G| = |P| - FP + FN.  [B, C-1].

    Hard counts + hard size give the hard Dice, soft counts + soft size the soft Dice."""
    tp = (size - fp).clamp_min(0)
    return 2 * tp / (2 * size - fp + fn).clamp_min(1e-6)


# --------------------------------------------------------------------------------------
# Lightning module
# --------------------------------------------------------------------------------------

class FPFNPredictionLightningModule(L.LightningModule):

    def __init__(
        self,
        wrapper: ScorePredictionWrapper,
        num_classes: int = 2,
        lr: float = 3.16e-5,
        patience: int = 20,
        loss: str = 'mse',
        count_target: str = 'hard',
        perturbations: str = 'predictor',
        adversarial_skip_epochs: int = 40,
        adversarial_warmup_epochs: int = 30,
        adv_target_dice_drop: float = 0.1,
        adv_controller_gain: float = 0.1,
        adv_fn_weight: float = 1.0,
        adv_mode: str = 'joint',
        adv_down: bool = False,
        adv_down_target_log_drop: float = 0.5,
        count_output: str = 'both',
        lr_monitor: str = 'val_loss',
        metadata: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['wrapper'])
        if loss not in ('mse', 'huber'):
            raise ValueError(f"loss must be 'mse' or 'huber', got {loss!r}")
        if adv_mode not in ADV_MODES:
            raise ValueError(f"adv_mode must be one of {ADV_MODES}, got {adv_mode!r}")
        if count_target not in COUNT_TARGETS:
            raise ValueError(f"count_target must be one of {COUNT_TARGETS}, got {count_target!r}")
        if perturbations not in ('none', 'predictor'):
            raise ValueError(f"perturbations must be 'none' or 'predictor', got {perturbations!r}")
        if count_output not in COUNT_OUTPUTS:
            raise ValueError(f"count_output must be one of {COUNT_OUTPUTS}, got {count_output!r}")
        if count_output != 'both' and adv_mode != 'joint':
            raise ValueError("a single-count head (count_output='fp'/'fn') has one adversary: adv_mode='joint'")
        if lr_monitor not in LR_MONITORS:
            raise ValueError(f"lr_monitor must be one of {LR_MONITORS}, got {lr_monitor!r}")
        self.wrapper = wrapper
        self.num_classes = int(num_classes)
        self.lr = lr
        self.patience = patience
        self.loss_name = loss
        self.count_target = count_target
        self.perturbations = perturbations
        self.adversarial_skip_epochs = int(adversarial_skip_epochs)
        self.adversarial_warmup_epochs = int(adversarial_warmup_epochs)
        self.adv_target_dice_drop = float(adv_target_dice_drop)
        self.adv_controller_gain = float(adv_controller_gain)
        self.adv_fn_weight = float(adv_fn_weight)
        self.adv_mode = adv_mode
        self.adv_down = bool(adv_down)
        self.adv_down_target_log_drop = float(adv_down_target_log_drop)
        self.count_output = count_output
        self.lr_monitor = lr_monitor
        self.metadata = metadata

        self.adv_gradient_factor = 1.0      # joint mode
        self.adv_fp_gradient_factor = 1.0   # split mode, FP half
        self.adv_fn_gradient_factor = 1.0   # split mode, FN half
        self.adv_down_gradient_factor = 1.0
        self._last_grad_norm = float('nan')
        self._last_fp_grad_norm = float('nan')
        self._last_fn_grad_norm = float('nan')
        self._last_down_grad_norm = float('nan')
        # adversarial copies of the previous step's clean batch, trained on in this step
        self.adversarial_batch: Optional[Dict[str, Tensor]] = None
        self.adversarial_down_batch: Optional[Dict[str, Tensor]] = None
        self.training_step_outputs: List[Dict] = []
        self.val_step_outputs: List[Dict] = []

    # -- persistence of the controller state --------------------------------------------

    _FACTORS = ('adv_gradient_factor', 'adv_fp_gradient_factor', 'adv_fn_gradient_factor',
                'adv_down_gradient_factor')

    def on_save_checkpoint(self, checkpoint: Dict) -> None:  # type: ignore[override]
        for k in self._FACTORS:
            checkpoint[k] = float(getattr(self, k))

    def on_load_checkpoint(self, checkpoint: Dict) -> None:  # type: ignore[override]
        for k in self._FACTORS:
            if k in checkpoint:
                setattr(self, k, float(checkpoint[k]))

    # -- helpers ------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        return self.wrapper(x)

    def regression_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Per-sample loss [B] in log(1 + count) space (only the trained count(s))."""
        pred = pred.float()
        if self.loss_name == 'huber':
            per_elem = F.smooth_l1_loss(pred, target, reduction='none', beta=1.0)
        else:
            per_elem = (pred - target).pow(2)
        if self.count_output != 'both':
            per_elem = per_elem[..., _CHANNEL[self.count_output]]
        return per_elem.flatten(1).mean(1)

    def own_error(self, log_counts: Tensor, fn_weight: Optional[float] = None) -> Tensor:
        """Per-sample [B] error the adversaries act on: log(1 + FP + a FN) for a joint head
        (a = `adv_fn_weight` unless `fn_weight` is given), log(1 + FP) resp. log(1 + FN) for a
        single-count head; mean over foreground classes."""
        if self.count_output == 'both':
            return log_error(log_counts, self.adv_fn_weight if fn_weight is None else fn_weight)
        return log_counts[..., _CHANNEL[self.count_output]].float().mean(1)

    def adversarial_active(self) -> bool:
        return self.perturbations == 'predictor' and self.current_epoch >= self.adversarial_skip_epochs

    def effective_step_factor(self) -> float:
        """0 during the skip epochs, then a linear ramp to 1 (same as the score predictor)."""
        if self.current_epoch < self.adversarial_skip_epochs:
            return 0.0
        if self.adversarial_warmup_epochs <= 0:
            return 1.0
        ramp = self.current_epoch - self.adversarial_skip_epochs
        return min(1.0, (ramp + 1) / self.adversarial_warmup_epochs)

    @staticmethod
    def _direction(err: Tensor, x_leaf: Tensor, rows: slice):
        """Per-sample normalized d err / d x for `rows` of `x_leaf`, and its mean L2 norm."""
        (grad,) = torch.autograd.grad(err, x_leaf, retain_graph=True)
        grad = torch.nan_to_num(grad[rows].float(), nan=0.0, posinf=0.0, neginf=0.0)
        norm = grad.flatten(1).norm(2, dim=1).view(-1, 1, 1, 1)
        return grad / (norm + 1e-8), float(norm.mean().item()) if norm.numel() else float('nan')

    def input_gradient(self, x: Tensor, pred: Tensor, x_leaf: Tensor):
        """Per-sample normalized d err / d x and its mean L2 norm, err = `own_error`
        (log(1 + FP_hat + a FN_hat), or the single count of an FP / FN head).

        `pred` must be the head output of the forward pass on `x_leaf` (requires grad);
        rows beyond x.size(0) (adversarial copies in the same batch) are ignored.
        """
        n = x.size(0)
        return self._direction(self.own_error(pred[:n]).sum(), x_leaf, slice(0, n))

    def adversarial_inputs(self, x: Tensor, pred: Tensor, x_leaf: Tensor):
        """(up, down, n_fp): steps that raise the predicted error and, with `adv_down`, one
        that lowers it like the TTA (`adv_down_gradient_factor`; else down = None).

        joint : all clean images along +d err (`adv_gradient_factor`), err = log(1 + FP_hat +
                a FN_hat), or log(1 + FP_hat) / log(1 + FN_hat) for a single-count head;
                n_fp = None.  `down` goes along the same direction, reversed.
        split : rows [:n_fp] (n_fp = n // 2) along +d log(1 + FP_hat) (`adv_fp_gradient_factor`),
                rows [n_fp:] along +d log(1 + FN_hat) (`adv_fn_gradient_factor`).  `down` keeps
                the joint direction (lowering the TTA's objective), on all rows.
        """
        n = x.size(0)
        xd = x.detach()
        joint = None
        if self.adv_mode == 'joint':
            n_fp = None
            joint, self._last_grad_norm = self.input_gradient(x, pred, x_leaf)
            up = xd + self.adv_gradient_factor * joint
        else:
            n_fp = n // 2
            fp_rows, fn_rows = slice(0, n_fp), slice(n_fp, n)
            # pred holds log(1 + count) per foreground class: [B, C-1, (fp, fn)]
            d_fp, self._last_fp_grad_norm = self._direction(pred[fp_rows, :, 0].sum(), x_leaf, fp_rows)
            d_fn, self._last_fn_grad_norm = self._direction(pred[fn_rows, :, 1].sum(), x_leaf, fn_rows)
            up = xd + torch.cat([self.adv_fp_gradient_factor * d_fp,
                                 self.adv_fn_gradient_factor * d_fn])
        down = None
        if self.adv_down:
            if joint is None:
                joint, self._last_down_grad_norm = self.input_gradient(x, pred, x_leaf)
            else:
                self._last_down_grad_norm = self._last_grad_norm
            down = (xd - self.adv_down_gradient_factor * joint).to(x.dtype).detach()
        return up.to(x.dtype).detach(), down, n_fp

    def _halves(self, n: int, n_fp: Optional[int]):
        """[(name, rows)] of the up adversary: [('', all)] (joint) or [('fp', ..), ('fn', ..)]."""
        if n_fp is None:
            return [('', slice(0, n))]
        return [('fp', slice(0, n_fp)), ('fn', slice(n_fp, n))]

    # -- training -----------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x = batch['input']
        t = batch['target'].long()
        t[t == -1] = 0
        n = x.size(0)
        use_adv = self.adversarial_active()
        adv = self.adversarial_batch if use_adv else None
        down = self.adversarial_down_batch if use_adv else None

        inputs, targets = [x], [t]
        if adv is not None:
            inputs.append(adv['input'])
            targets.append(adv['target'])
        if down is not None:
            inputs.append(down['input'])
            targets.append(down['target'])
        n_up = adv['input'].size(0) if adv is not None else 0
        up_rows = slice(n, n + n_up)
        down_rows = slice(n + n_up, None)
        x_all = torch.cat(inputs).detach().requires_grad_(use_adv)
        t_all = torch.cat(targets)

        logits = self(x_all)
        pred = self.wrapper.prediction                                          # [B, C-1, 2]
        tgt, _ = fpfn_targets(logits, t_all, self.num_classes, self.count_target)  # [B, C-1, 2]
        dice = hard_dice(logits, t_all, self.num_classes)                       # [B]
        per_sample = self.regression_loss(pred, tgt)
        loss = per_sample.mean()

        metrics = {
            'loss': loss.detach().view(1).cpu(),
            'loss_clean': per_sample[:n].detach().mean().view(1).cpu(),
        }
        if adv is not None:
            metrics['loss_adv'] = per_sample[up_rows].detach().mean().view(1).cpu()
            # controller (as for the score predictor): did the TRUE hard Dice of the
            # adversarial copies drop by the target, relative to their clean originals?
            # split mode: one controller per half (FP / FN adversary), same target.
            drop = adv['dice'] - dice[up_rows]                       # [n_up] true Dice drop
            dlog = (tgt[up_rows] - adv['true_log_counts']).float()   # [n_up, C-1, 2] rise of log(1+FP|FN)
            target_delta = self.adv_target_dice_drop * self.effective_step_factor()
            metrics['adv_observed_delta'] = drop.detach().mean().view(1).cpu()
            for name, rows in self._halves(n_up, adv['n_fp']):
                if drop[rows].numel() == 0:
                    continue
                tag = f'adv_{name}_' if name else 'adv_'
                observed = drop[rows].mean()
                step_error = float(observed.item()) - target_delta
                factor = getattr(self, f'{tag}gradient_factor') - self.adv_controller_gain * step_error
                setattr(self, f'{tag}gradient_factor', float(max(1e-4, min(10.0, factor))))
                if name:
                    metrics[f'{tag}observed_delta'] = observed.detach().view(1).cpu()
                metrics[f'{tag}step_error'] = torch.tensor([step_error])
                metrics[f'{tag}true_dlog_fp'] = dlog[rows, :, 0].mean().view(1).cpu()
                metrics[f'{tag}true_dlog_fn'] = dlog[rows, :, 1].mean().view(1).cpu()
        if down is not None:
            metrics['loss_adv_down'] = per_sample[down_rows].detach().mean().view(1).cpu()
            # controller on the PREDICTED error (the true one should not move): did the
            # head's log(1 + FP_hat + a FN_hat) drop by the target relative to the clean
            # originals?  Like the score predictor's concentration adversary.
            err_down = self.own_error(pred[down_rows].detach())
            observed_down = (down['pred_log_error'] - err_down).mean()
            target_down = self.adv_down_target_log_drop * self.effective_step_factor()
            down_error = float(observed_down.item()) - target_down
            self.adv_down_gradient_factor -= self.adv_controller_gain * down_error
            self.adv_down_gradient_factor = float(max(1e-4, min(10.0, self.adv_down_gradient_factor)))
            true_down = (self.own_error(down['true_log_counts'], 1.0) - self.own_error(tgt[down_rows], 1.0)).mean()
            metrics['adv_down_pred_drop'] = observed_down.detach().view(1).cpu()
            metrics['adv_down_true_drop'] = true_down.detach().view(1).cpu()
            metrics['adv_down_dice_gain'] = (dice[down_rows] - down['dice']).mean().detach().view(1).cpu()

        if use_adv:
            adv_x, down_x, n_fp = self.adversarial_inputs(x, pred, x_all)
            self.adversarial_batch = {
                'input': adv_x,
                'target': t.detach(),
                'dice': dice[:n].detach(),
                'true_log_counts': tgt[:n].detach(),
                'n_fp': n_fp,   # split mode: rows [:n_fp] FP adversary, rest FN; None = joint
            }
            if down_x is not None:
                self.adversarial_down_batch = {
                    'input': down_x,
                    'target': t.detach(),
                    'dice': dice[:n].detach(),
                    'pred_log_error': self.own_error(pred[:n].detach()),
                    'true_log_counts': tgt[:n].detach(),
                }
                metrics['adv_down_gradient_factor'] = torch.tensor([self.adv_down_gradient_factor])
            if n_fp is None:
                metrics['adv_gradient_factor'] = torch.tensor([self.adv_gradient_factor])
                metrics['adv_grad_norm'] = torch.tensor([self._last_grad_norm])
            else:
                for name in ('fp', 'fn'):
                    metrics[f'adv_{name}_gradient_factor'] = torch.tensor([getattr(self, f'adv_{name}_gradient_factor')])
                    metrics[f'adv_{name}_grad_norm'] = torch.tensor([getattr(self, f'_last_{name}_grad_norm')])

        self.training_step_outputs.append(metrics)
        return loss

    def on_train_epoch_end(self):
        outs = self.training_step_outputs
        keys = set().union(*[o.keys() for o in outs]) if outs else set()
        for key in sorted(keys):
            vals = torch.cat([o[key] for o in outs if key in o])
            name = 'train_loss' if key == 'loss' else f'train_{key}'
            self.log(name, vals.mean(), on_epoch=True, prog_bar=key == 'loss', logger=True)
        self.log('effective_step_factor', self.effective_step_factor(), on_epoch=True, logger=True)
        # plain-text trace for log files (no wandb)
        mean = lambda k: torch.cat([o[k] for o in outs if k in o]).float().nanmean().item() if k in keys else float('nan')
        target_txt = f"(target {self.adv_target_dice_drop * self.effective_step_factor():.3f})"
        if 'adv_gradient_factor' in keys:
            print(f"[epoch {self.current_epoch}] loss clean {mean('loss_clean'):.3f} adv {mean('loss_adv'):.3f}"
                  f" | adv factor {self.adv_gradient_factor:.3f} grad norm {mean('adv_grad_norm'):.2e}"
                  f" true Dice drop {mean('adv_observed_delta'):+.3f} {target_txt}"
                  f" true dlog FP {mean('adv_true_dlog_fp'):+.3f} FN {mean('adv_true_dlog_fn'):+.3f}")
        if 'adv_fp_gradient_factor' in keys:
            print(f"[epoch {self.current_epoch}] loss clean {mean('loss_clean'):.3f} adv {mean('loss_adv'):.3f} {target_txt}")
            for name in ('fp', 'fn'):
                print(f"[epoch {self.current_epoch}]   {name.upper()} adversary: factor"
                      f" {getattr(self, f'adv_{name}_gradient_factor'):.3f}"
                      f" grad norm {mean(f'adv_{name}_grad_norm'):.2e}"
                      f" true Dice drop {mean(f'adv_{name}_observed_delta'):+.3f}"
                      f" true dlog FP {mean(f'adv_{name}_true_dlog_fp'):+.3f}"
                      f" FN {mean(f'adv_{name}_true_dlog_fn'):+.3f}")
        if 'adv_down_gradient_factor' in keys:
            print(f"[epoch {self.current_epoch}] down: loss {mean('loss_adv_down'):.3f}"
                  f" | factor {self.adv_down_gradient_factor:.3f}"
                  f" pred log-err drop {mean('adv_down_pred_drop'):+.3f}"
                  f" (target {self.adv_down_target_log_drop * self.effective_step_factor():.3f})"
                  f" true log-err drop {mean('adv_down_true_drop'):+.3f}"
                  f" true Dice gain {mean('adv_down_dice_gain'):+.3f}")
        self.training_step_outputs.clear()

    # -- evaluation ---------------------------------------------------------------------

    @torch.enable_grad()
    def eval_step(self, batch, adversarial: bool) -> Dict[str, Tensor]:
        """Per-sample predictions and targets; optionally also for one adversarial step."""
        x = batch['input']
        t = batch['target'].long()
        t[t == -1] = 0
        x_leaf = x.detach().requires_grad_(adversarial)
        logits = self(x_leaf)
        pred = self.wrapper.prediction
        tgt, size = fpfn_targets(logits, t, self.num_classes, self.count_target)
        dice = hard_dice(logits, t, self.num_classes)
        out = {
            'pred': pred.detach().float().cpu(),
            'target': tgt.cpu(),
            'size': size.cpu(),          # [B, C-1] |P|, hard or soft as the counts
            'dice': dice.cpu(),          # [B] true hard Dice
            'loss': self.regression_loss(pred, tgt).detach().cpu(),
        }
        if adversarial:
            adv_x, down_x, n_fp = self.adversarial_inputs(x, pred, x_leaf)
            kind = torch.full((x.size(0),), ADV_KIND['joint'], dtype=torch.long)
            if n_fp is not None:
                kind[:n_fp], kind[n_fp:] = ADV_KIND['fp'], ADV_KIND['fn']
            if down_x is not None:
                with torch.no_grad():
                    logits_d = self(down_x)
                    pred_d = self.wrapper.prediction
                    tgt_d, _ = fpfn_targets(logits_d, t, self.num_classes, self.count_target)
                    dice_d = hard_dice(logits_d, t, self.num_classes)
                out.update({
                    'adv_down_loss': self.regression_loss(pred_d, tgt_d).cpu(),
                    'adv_down_pred_drop': (self.own_error(pred.detach())
                                           - self.own_error(pred_d)).float().cpu(),
                    'adv_down_true_drop': (self.own_error(tgt, 1.0) - self.own_error(tgt_d, 1.0)).float().cpu(),
                    'adv_down_dice_gain': (dice_d - dice).cpu(),
                })
            with torch.no_grad():
                logits_a = self(adv_x)
                pred_a = self.wrapper.prediction
                tgt_a, _ = fpfn_targets(logits_a, t, self.num_classes, self.count_target)
                dice_a = hard_dice(logits_a, t, self.num_classes)
            out.update({
                'adv_pred': pred_a.float().cpu(),
                'adv_target': tgt_a.cpu(),
                'adv_dice_drop': (dice - dice_a).cpu(),
                'adv_loss': self.regression_loss(pred_a, tgt_a).cpu(),
                'adv_kind': kind,                                  # ADV_KIND per sample
                'adv_true_dlog': (tgt_a - tgt).float().cpu(),      # [B, C-1, 2] rise of log(1+FP|FN)
            })
        return out

    def validation_step(self, batch, batch_idx):
        self.val_step_outputs.append(self.eval_step(batch, adversarial=self.adversarial_active()))

    def on_validation_epoch_end(self):
        outs = {k: torch.cat([o[k] for o in self.val_step_outputs]) for k in self.val_step_outputs[0]}
        # same structure as the score predictor: clean loss (+ adversarial loss once active)
        val_loss = outs['loss'].mean()
        if 'adv_loss' in outs and self.current_epoch >= self.adversarial_skip_epochs + 1:
            val_loss = val_loss + outs['adv_loss'].mean()
            if 'adv_down_loss' in outs:
                val_loss = val_loss + outs['adv_down_loss'].mean()
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss_clean', outs['loss'].mean(), on_epoch=True, logger=True)
        if 'adv_loss' in outs:
            self.log('val_loss_adv', outs['adv_loss'].mean(), on_epoch=True, logger=True)
        for k in ('adv_down_loss', 'adv_down_pred_drop', 'adv_down_true_drop', 'adv_down_dice_gain'):
            if k in outs:
                self.log(f'val_{k}', outs[k].float().mean(), on_epoch=True, logger=True)
        if 'adv_kind' in outs:  # per up adversary: true Dice drop and rise of the true log counts
            for name, code in ADV_KIND.items():
                sel = outs['adv_kind'] == code
                if not sel.any():
                    continue
                tag = 'val_adv' if name == 'joint' else f'val_adv_{name}'
                self.log(f'{tag}_dice_drop', outs['adv_dice_drop'][sel].float().mean(), on_epoch=True, logger=True)
                self.log(f'{tag}_true_dlog_fp', outs['adv_true_dlog'][sel][..., 0].mean(), on_epoch=True, logger=True)
                self.log(f'{tag}_true_dlog_fn', outs['adv_true_dlog'][sel][..., 1].mean(), on_epoch=True, logger=True)
        for name, val in summary_metrics(outs).items():
            self.log(f'val_{name}', val, on_epoch=True, logger=True)
        self.val_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self.eval_step(batch, adversarial=False)

    def configure_optimizers(self):
        optimizer = Adam(self.wrapper.prediction_head.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=self.patience),
                'monitor': self.lr_monitor,
                'frequency': 1,
            },
        }


def _corr(a: Tensor, b: Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    if a.numel() < 2 or a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def summary_metrics(outs: Dict[str, Tensor]) -> Dict[str, float]:
    """MAE / Pearson r in log space for FP, FN and the joint error, and for the Dice implied
    by the predicted counts (vs. the Dice implied by the true counts: the hard Dice for
    hard counts, the soft Dice for soft counts).  `outs` as from `eval_step`.

    NOTE the implied Dice uses the true mask size |P| (known from the logits), so even a
    constant FP/FN prediction correlates with the true Dice: compare `corr_dice` with that
    of mu-hat, not with 0."""
    pred, tgt = outs['pred'], outs['target']
    m = {
        'mae_log_fp': float((pred[..., 0] - tgt[..., 0]).abs().mean()),
        'mae_log_fn': float((pred[..., 1] - tgt[..., 1]).abs().mean()),
        'corr_log_fp': _corr(pred[..., 0], tgt[..., 0]),
        'corr_log_fn': _corr(pred[..., 1], tgt[..., 1]),
        'corr_log_error': _corr(log_error(pred), log_error(tgt)),
    }
    s = outs['size']
    dice_hat = dice_from_counts(s, *log_counts_to_counts(pred)).mean(1)
    dice_true = dice_from_counts(s, *log_counts_to_counts(tgt)).mean(1)
    m['mae_dice'] = float((dice_hat - dice_true).abs().mean())
    m['corr_dice'] = _corr(dice_hat, dice_true)
    return m


# --------------------------------------------------------------------------------------
# construction / loading / attaching
# --------------------------------------------------------------------------------------

def get_fpfn_prediction_module(
    data_cfg,
    model_cfg,
    unet: nn.Module,
    metadata: Optional[Dict] = None,
    ckpt: Optional[str] = None,
) -> FPFNPredictionLightningModule:
    """Build (ckpt=None) or load the FP/FN head on top of a frozen copy of `unet`."""
    swivels = resolve_score_swivels(unet, model_cfg.swivels)
    output_shapes = find_shapes_for_swivels(
        model=unet, swivels=swivels, input_shape=OmegaConf.to_object(data_cfg.input_shape)
    )
    head = FPFNPredictionHead(output_shapes=output_shapes,
                              num_foreground=int(model_cfg.num_classes) - 1)
    wrapper = ScorePredictionWrapper(
        model=unet,
        prediction_head=head,
        adapters=nn.ModuleList([ScorePredictionAdapter(swivel=s) for s in swivels]),
        placement=getattr(model_cfg, 'hook_placement', 'pre'),
    )
    wrapper.freeze_normalization_layers()
    wrapper.freeze_model()

    kwargs = dict(
        num_classes=int(model_cfg.num_classes),
        lr=float(model_cfg.lr),
        patience=int(model_cfg.patience),
        loss=str(getattr(model_cfg, 'regression_loss', 'mse')),
        count_target=str(getattr(model_cfg, 'count_target', 'hard')),
        perturbations=str(model_cfg.perturbations),
        adversarial_skip_epochs=int(model_cfg.adversarial_skip_epochs),
        adversarial_warmup_epochs=int(model_cfg.adversarial_warmup_epochs),
        adv_target_dice_drop=float(model_cfg.adv_target_dice_drop),
        adv_controller_gain=float(getattr(model_cfg, 'adv_controller_gain', 0.1)),
        adv_fn_weight=float(getattr(model_cfg, 'adv_fn_weight', 1.0)),
        adv_mode=str(getattr(model_cfg, 'adv_mode', 'joint')),  # old heads: joint
        adv_down=bool(getattr(model_cfg, 'adv_down', False)),
        adv_down_target_log_drop=float(getattr(model_cfg, 'adv_down_target_log_drop', 0.5)),
        count_output=str(getattr(model_cfg, 'count_output', 'both')),   # old heads: both
        lr_monitor=str(getattr(model_cfg, 'lr_monitor', 'val_loss')),
        metadata=metadata,
    )
    if ckpt is None:
        return FPFNPredictionLightningModule(wrapper=wrapper, **kwargs)
    return FPFNPredictionLightningModule.load_from_checkpoint(
        checkpoint_path=ckpt, wrapper=wrapper, **kwargs
    )


def load_fpfn_head(ckpt_path: str, data_cfg, unet: nn.Module, device: str = 'cuda:0'):
    """Load a trained FP/FN head; its model config is taken from the checkpoint metadata."""
    meta = (
        torch.load(ckpt_path, map_location='cpu', weights_only=False)
        .get('hyper_parameters', {})
        .get('metadata')
        or {}
    )
    if not meta:
        raise ValueError(f'{ckpt_path} carries no metadata; was it trained by train_fpfn_head.py?')
    model_cfg = OmegaConf.create(meta)
    module = get_fpfn_prediction_module(data_cfg, model_cfg, unet, metadata=meta, ckpt=ckpt_path)
    module = module.to(device).eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module, meta


class CombinedFPFNHead(nn.Module):
    """FP from an FP-only head and FN from an FN-only head: [B, C-1, 2] like a joint head."""

    def __init__(self, fp_head: nn.Module, fn_head: nn.Module):
        super().__init__()
        self.fp_head, self.fn_head = fp_head, fn_head

    def forward(self, feats: dict) -> Tensor:
        # the feature fuser re-keys its input dict in place: every head gets its own copy
        fp = self.fp_head(dict(feats))[..., 0]
        fn = self.fn_head(dict(feats))[..., 1]
        return torch.stack([fp, fn], dim=-1)


def _check_compatible(wrapper: ScorePredictionWrapper, module: FPFNPredictionLightningModule) -> None:
    own = [a.swivel for a in module.wrapper.adapters]
    host = [a.swivel for a in wrapper.adapters]
    if own != host:
        raise ValueError(f'FP/FN head reads swivels {own}, the score predictor {host}')
    if module.wrapper.placement != wrapper.placement:
        raise ValueError(
            f'FP/FN head trained with hook placement {module.wrapper.placement!r}, '
            f'score predictor uses {wrapper.placement!r}'
        )


def attach_fpfn_head(wrapper: ScorePredictionWrapper, module: FPFNPredictionLightningModule,
                     fn_module: Optional[FPFNPredictionLightningModule] = None) -> None:
    """Let the FP/FN head read the features of `wrapper` (the score predictor's wrapper).

    Both must record the same swivels with the same placement, otherwise the head would
    receive features it was not trained on.  After every `wrapper(x)`,
    `wrapper.extra_predictions[FPFN_HEAD]` holds [B, C-1, 2] = (log(1+FP_hat), log(1+FN_hat)).

    `module` alone must be a joint head (count_output='both').  For separate heads pass the
    FP-only head as `module` and the FN-only head as `fn_module`.
    """
    _check_compatible(wrapper, module)
    if fn_module is None:
        if module.count_output != 'both':
            raise ValueError(f"head trained on {module.count_output!r} only; pass the "
                             f"{'FN' if module.count_output == 'fp' else 'FP'} head as fn_module / module")
        head = module.wrapper.prediction_head
    else:
        _check_compatible(wrapper, fn_module)
        if (module.count_output, fn_module.count_output) != ('fp', 'fn'):
            raise ValueError(f"separate heads must be (fp, fn), got "
                             f"({module.count_output!r}, {fn_module.count_output!r})")
        if module.count_target != fn_module.count_target:
            raise ValueError(f'FP head counts {module.count_target!r}, FN head {fn_module.count_target!r}')
        head = CombinedFPFNHead(module.wrapper.prediction_head, fn_module.wrapper.prediction_head)
    head.eval()
    wrapper.extra_heads[FPFN_HEAD] = head


def find_fpfn_head_checkpoint(base_dir: str, dataset: str, split: str, name: str) -> str:
    pattern = os.path.join(base_dir, f'{dataset}_{split}_*{name}_*.ckpt')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No FP/FN head checkpoint matching {pattern}')
    if len(matches) > 1:
        print(f'[info] {len(matches)} FP/FN head checkpoints matched; using {os.path.basename(matches[0])}')
    return matches[0]
