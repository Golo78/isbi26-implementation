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
plausible, unverified cause of the ~5x FN underestimation under shift.  Open alternative:
one FP and one FN adversary with B/2 images each, each with its own controller.

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
        adv_down: bool = False,
        adv_down_target_log_drop: float = 0.5,
        metadata: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['wrapper'])
        if loss not in ('mse', 'huber'):
            raise ValueError(f"loss must be 'mse' or 'huber', got {loss!r}")
        if count_target not in COUNT_TARGETS:
            raise ValueError(f"count_target must be one of {COUNT_TARGETS}, got {count_target!r}")
        if perturbations not in ('none', 'predictor'):
            raise ValueError(f"perturbations must be 'none' or 'predictor', got {perturbations!r}")
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
        self.adv_down = bool(adv_down)
        self.adv_down_target_log_drop = float(adv_down_target_log_drop)
        self.metadata = metadata

        self.adv_gradient_factor = 1.0
        self.adv_down_gradient_factor = 1.0
        self._last_grad_norm = float('nan')
        self._last_down_grad_norm = float('nan')
        # adversarial copies of the previous step's clean batch, trained on in this step
        self.adversarial_batch: Optional[Dict[str, Tensor]] = None
        self.adversarial_down_batch: Optional[Dict[str, Tensor]] = None
        self.training_step_outputs: List[Dict] = []
        self.val_step_outputs: List[Dict] = []

    # -- persistence of the controller state --------------------------------------------

    def on_save_checkpoint(self, checkpoint: Dict) -> None:  # type: ignore[override]
        checkpoint['adv_gradient_factor'] = float(self.adv_gradient_factor)
        checkpoint['adv_down_gradient_factor'] = float(self.adv_down_gradient_factor)

    def on_load_checkpoint(self, checkpoint: Dict) -> None:  # type: ignore[override]
        if 'adv_gradient_factor' in checkpoint:
            self.adv_gradient_factor = float(checkpoint['adv_gradient_factor'])
        if 'adv_down_gradient_factor' in checkpoint:
            self.adv_down_gradient_factor = float(checkpoint['adv_down_gradient_factor'])

    # -- helpers ------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        return self.wrapper(x)

    def regression_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Per-sample loss [B] in log(1 + count) space."""
        pred = pred.float()
        if self.loss_name == 'huber':
            per_elem = F.smooth_l1_loss(pred, target, reduction='none', beta=1.0)
        else:
            per_elem = (pred - target).pow(2)
        return per_elem.flatten(1).mean(1)

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

    def input_gradient(self, x: Tensor, pred: Tensor, x_leaf: Tensor):
        """Per-sample normalized d log(1 + FP_hat + a FN_hat) / d x and its mean L2 norm.

        `pred` must be the head output of the forward pass on `x_leaf` (requires grad);
        rows beyond x.size(0) (adversarial copies in the same batch) are ignored.
        """
        n = x.size(0)
        err = log_error(pred[:n], self.adv_fn_weight).sum()
        (grad,) = torch.autograd.grad(err, x_leaf, retain_graph=True)
        grad = torch.nan_to_num(grad[:n].float(), nan=0.0, posinf=0.0, neginf=0.0)
        norm = grad.flatten(1).norm(2, dim=1).view(-1, 1, 1, 1)
        return grad / (norm + 1e-8), float(norm.mean().item())

    def adversarial_input(self, x: Tensor, pred: Tensor, x_leaf: Tensor, factor: float) -> Tensor:
        """Move `x` one normalized step along +d log(1 + FP_hat + a FN_hat) / d x."""
        direction, self._last_grad_norm = self.input_gradient(x, pred, x_leaf)
        return (x.detach() + factor * direction).to(x.dtype).detach()

    def adversarial_inputs(self, x: Tensor, pred: Tensor, x_leaf: Tensor):
        """(up, down): one step that raises the predicted error (step `adv_gradient_factor`)
        and, with `adv_down`, one that lowers it like the TTA (`adv_down_gradient_factor`;
        else None).  Both along the same normalized gradient of the clean images."""
        direction, norm = self.input_gradient(x, pred, x_leaf)
        self._last_grad_norm = self._last_down_grad_norm = norm
        up = (x.detach() + self.adv_gradient_factor * direction).to(x.dtype).detach()
        down = None
        if self.adv_down:
            down = (x.detach() - self.adv_down_gradient_factor * direction).to(x.dtype).detach()
        return up, down

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
            observed = (adv['dice'] - dice[up_rows]).mean()
            target_delta = self.adv_target_dice_drop * self.effective_step_factor()
            step_error = float(observed.item()) - target_delta
            self.adv_gradient_factor -= self.adv_controller_gain * step_error
            self.adv_gradient_factor = float(max(1e-4, min(10.0, self.adv_gradient_factor)))
            metrics['adv_observed_delta'] = observed.detach().view(1).cpu()
            metrics['adv_step_error'] = torch.tensor([step_error])
        if down is not None:
            metrics['loss_adv_down'] = per_sample[down_rows].detach().mean().view(1).cpu()
            # controller on the PREDICTED error (the true one should not move): did the
            # head's log(1 + FP_hat + a FN_hat) drop by the target relative to the clean
            # originals?  Like the score predictor's concentration adversary.
            err_down = log_error(pred[down_rows].detach(), self.adv_fn_weight)
            observed_down = (down['pred_log_error'] - err_down).mean()
            target_down = self.adv_down_target_log_drop * self.effective_step_factor()
            down_error = float(observed_down.item()) - target_down
            self.adv_down_gradient_factor -= self.adv_controller_gain * down_error
            self.adv_down_gradient_factor = float(max(1e-4, min(10.0, self.adv_down_gradient_factor)))
            true_down = (log_error(down['true_log_counts']) - log_error(tgt[down_rows])).mean()
            metrics['adv_down_pred_drop'] = observed_down.detach().view(1).cpu()
            metrics['adv_down_true_drop'] = true_down.detach().view(1).cpu()
            metrics['adv_down_dice_gain'] = (dice[down_rows] - down['dice']).mean().detach().view(1).cpu()

        if use_adv:
            adv_x, down_x = self.adversarial_inputs(x, pred, x_all)
            self.adversarial_batch = {
                'input': adv_x,
                'target': t.detach(),
                'dice': dice[:n].detach(),
            }
            if down_x is not None:
                self.adversarial_down_batch = {
                    'input': down_x,
                    'target': t.detach(),
                    'dice': dice[:n].detach(),
                    'pred_log_error': log_error(pred[:n].detach(), self.adv_fn_weight),
                    'true_log_counts': tgt[:n].detach(),
                }
                metrics['adv_down_gradient_factor'] = torch.tensor([self.adv_down_gradient_factor])
            metrics['adv_gradient_factor'] = torch.tensor([self.adv_gradient_factor])
            metrics['adv_grad_norm'] = torch.tensor([self._last_grad_norm])

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
        if 'adv_gradient_factor' in keys:  # plain-text trace for log files (no wandb)
            mean = lambda k: torch.cat([o[k] for o in outs if k in o]).mean().item() if k in keys else float('nan')
            print(f"[epoch {self.current_epoch}] loss clean {mean('loss_clean'):.3f} adv {mean('loss_adv'):.3f}"
                  f" | adv factor {self.adv_gradient_factor:.3f} grad norm {mean('adv_grad_norm'):.2e}"
                  f" true Dice drop {mean('adv_observed_delta'):+.3f}"
                  f" (target {self.adv_target_dice_drop * self.effective_step_factor():.3f})")
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
            adv_x, down_x = self.adversarial_inputs(x, pred, x_leaf)
            if down_x is not None:
                with torch.no_grad():
                    logits_d = self(down_x)
                    pred_d = self.wrapper.prediction
                    tgt_d, _ = fpfn_targets(logits_d, t, self.num_classes, self.count_target)
                    dice_d = hard_dice(logits_d, t, self.num_classes)
                out.update({
                    'adv_down_loss': self.regression_loss(pred_d, tgt_d).cpu(),
                    'adv_down_pred_drop': (log_error(pred.detach(), self.adv_fn_weight)
                                           - log_error(pred_d, self.adv_fn_weight)).float().cpu(),
                    'adv_down_true_drop': (log_error(tgt) - log_error(tgt_d)).float().cpu(),
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
                'monitor': 'val_loss',
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
        adv_down=bool(getattr(model_cfg, 'adv_down', False)),
        adv_down_target_log_drop=float(getattr(model_cfg, 'adv_down_target_log_drop', 0.5)),
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


def attach_fpfn_head(wrapper: ScorePredictionWrapper, module: FPFNPredictionLightningModule) -> None:
    """Let the FP/FN head read the features of `wrapper` (the score predictor's wrapper).

    Both must record the same swivels with the same placement, otherwise the head would
    receive features it was not trained on.  After every `wrapper(x)`,
    `wrapper.extra_predictions[FPFN_HEAD]` holds [B, C-1, 2] = (log(1+FP_hat), log(1+FN_hat)).
    """
    own = [a.swivel for a in module.wrapper.adapters]
    host = [a.swivel for a in wrapper.adapters]
    if own != host:
        raise ValueError(f'FP/FN head reads swivels {own}, the score predictor {host}')
    if module.wrapper.placement != wrapper.placement:
        raise ValueError(
            f'FP/FN head trained with hook placement {module.wrapper.placement!r}, '
            f'score predictor uses {wrapper.placement!r}'
        )
    head = module.wrapper.prediction_head
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
