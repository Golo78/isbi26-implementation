"""Latent-space test-time adaptation driven by the confidence predictor.

Instead of perturbing the *input* (as in the MIDL adversarial training scheme), we keep
the image fixed and adapt a small, spatially-invariant transform of the *latent features*
at a subset of the score-predictor swivels, using gradient ascent on the predicted score
mu-hat.  The goal is a net improvement of the true Dice on domain-shifted data.

IMPORTANT -- placement
----------------------
The swivels used by `ScorePredictionWrapper` are `adn.N` modules, i.e.
`InstanceNorm2d(affine=False, track_running_stats=False)`, and the score adapters are
attached with `register_forward_pre_hook`.  A per-channel affine applied at that position
is *annihilated* by the normalization that immediately follows it:

    IN(gamma_c * x + beta_c) == IN(x)      (exactly, up to eps and sign(gamma))

Empirically, a pre-norm affine leaves mu-hat and Dice bit-for-bit unchanged and yields
d mu-hat / d gamma == 0.  Adapters here are therefore registered as *forward* hooks and
act on the normalization's OUTPUT, which is equivalent to giving those InstanceNorms the
affine parameters they were constructed without.

Recommended adaptation sites are indices [0, 17] of the `adn.N` modules (first encoder
conv and penultimate decoder level).  Index -1 (`model.2.0.adn.N`) feeds the logits and
adapting it raises mu-hat while *lowering* Dice, so it is excluded by default.

`placement='pre'` (`attach_latent_adapters`) exists to *measure* the effect described
above instead of asserting it: the adapter then runs as a forward PRE hook, prepended so
that it fires before the `ScorePredictionAdapter` pre-hook and the head therefore reads
the adapted features.  The segmentation path, however, still sees the affine annihilated
by the InstanceNorm, so the expected signature is "mu-hat moves, logits do not".  The
per-step `logit_drift` in the trace quantifies exactly that.

Objectives (`run_latent_tta`)
-----------------------------
    'mu'          : ascend the predicted score mu-hat.  Deployable, label-free.
    'oracle-dice' : ascend the *true* soft Dice.  Needs labels, is NOT deployable, and
                    serves as the upper bound of what latent adaptation could achieve if
                    the quality estimate were perfect.  Comparing the two isolates how
                    much of the gain hinges on the mu-hat / true-Dice correlation.

Independently of the objective, `kappa_weight > 0` adds a simultaneous descent on the
predictive uncertainty of the QC head (Beta sigma, or -log kappa).

`size_weight > 0` adds a simultaneous descent on the error of the soft foreground size
against a per-image target size `size_target` that is fixed BEFORE the adaptation (in
`apps/eval_latent_tta.py`: G_hat = |P_0| - FP_0 + FN_0 from the true FP/FN counts of the
un-adapted mask, optionally noised -- an oracle for a head that predicts FP and FN).
The term carries no location information: mu-hat decides *where* the mask changes, the
size term *how much*.

`fpfn_weight > 0` instead descends the TRUE soft FP and FN counts of the current mask at
every step, anchored at 0 (`fpfn_fn`, labels) -- an oracle for a head that predicts FP and
FN perfectly for every mask the adaptation produces.  `fpfn_source='head'` replaces the true counts by the prediction of a trained
FP/FN head (`model/fpfn_head.py`, attached as `wrapper.extra_heads['fpfn']`) -- the
deployable, label-free version of the same term.  `fpfn_form='separate'` descends
log(1 + FP) + log(1 + FN) = log((1 + FP)(1 + FN)), whose minimum is reached as soon as ONE of
the two counts is 0 (e.g. an under-segmenting mask with FP = 0); `fpfn_form='joint'`
descends log(1 + FP + FN), which only vanishes for a perfect mask.  `fpfn_fn_weight` scales
FN inside either form (> 1: a missing foreground pixel costs more than a wrong one).

Logit adapter (`adapt_logits=True`)
-----------------------------------
Adds one more per-sample, per-channel affine directly on the UNet OUTPUT, registered as a
forward hook on `wrapper.model` itself.  `ScorePredictionWrapper.forward` uses that output
both as the returned segmentation logits and as the head's last swivel, so segmentation and
QC head see the same adapted logits.  For a binary (1-channel) network the hard mask is

    exp(log_gamma) * z + beta > 0   <=>   z > -beta * exp(-log_gamma)

i.e. on the hard segmentation a logit affine is *exactly* a per-image global threshold
shift (gamma only acts as a temperature on the soft logits the head reads).  The
threshold-shift oracle in `apps/eval_latent_tta.py` is therefore the label-informed
optimum of what the logit adapter alone could do to the hard mask.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

import sys
sys.path.append('../')
from utils import find_shapes_for_swivels
from losses import soft_size_from_logits
from model.fpfn_head import FPFN_HEAD


# --------------------------------------------------------------------------------------
# swivel resolution
# --------------------------------------------------------------------------------------

def resolve_swivels(model: nn.Module, indices: Sequence[int], match: str = 'adn.N') -> List[str]:
    """Map integer indices to module paths, matching `get_score_prediction_module`.

    Kept here so the training path and the TTA path cannot drift apart on which layers
    'the attachment points' are.
    """
    names = [name for name, _ in model.named_modules() if match in name]
    print(f"[info] {len(names)} swivels matching '{match}' found in model")
    print(f"[info] swivel names: {names}")
    if not names:
        raise ValueError(f"No modules matching '{match}' in model")
    out = []
    for i in indices:
        i = int(i)
        if not -len(names) <= i < len(names):
            raise IndexError(f'swivel index {i} out of range for {len(names)} matches')
        out.append(names[i])
    return out


# --------------------------------------------------------------------------------------
# adapter
# --------------------------------------------------------------------------------------

class LatentAffineAdapter(nn.Module):
    """Per-sample, per-channel affine on the OUTPUT of an InstanceNorm swivel.

    Parameters are shaped [N, C, 1, 1] so that a single adapter can serve a *mixed*
    batch: rows left at zero are exactly the identity (exp(0)*x + 0), which makes it safe
    to keep the hooks attached during ordinary training and only activate the rows that
    belong to an adapted chunk.

    The parameters are deliberately NOT registered on the owning LightningModule (they are
    held in a plain list, see `attach_latent_adapters`) so that they
      * never enter the head optimizer or the checkpoint state_dict, and
      * are not wiped by `LightningModule.zero_grad()`.
    """

    def __init__(self, swivel: str, channels: int, max_batch: int, device: str = 'cuda:0',
                 placement: str = 'post'):
        super().__init__()
        self.swivel = swivel
        self.placement = placement  # 'post' (norm output) | 'pre' (norm input)
        self.channels = int(channels)
        self.max_batch = int(max_batch)
        self.device_str = device
        self.active = True
        self.log_gamma = nn.Parameter(
            torch.zeros(self.max_batch, self.channels, 1, 1, device=device)
        )
        self.beta = nn.Parameter(
            torch.zeros(self.max_batch, self.channels, 1, 1, device=device)
        )

    # -- state management ---------------------------------------------------------------

    def parameters_list(self) -> List[nn.Parameter]:
        return [self.log_gamma, self.beta]

    @torch.no_grad()
    def reset(self, rows: Optional[slice] = None) -> None:
        if rows is None:
            self.log_gamma.zero_()
            self.beta.zero_()
        else:
            self.log_gamma[rows] = 0.0
            self.beta[rows] = 0.0

    @torch.no_grad()
    def load_rows(self, rows: slice, state: Tuple[Tensor, Tensor]) -> None:
        lg, b = state
        self.log_gamma[rows] = lg.to(self.log_gamma.device)
        self.beta[rows] = b.to(self.beta.device)

    @torch.no_grad()
    def export_rows(self, rows: slice) -> Tuple[Tensor, Tensor]:
        return self.log_gamma[rows].detach().clone(), self.beta[rows].detach().clone()

    def zero_grad_(self) -> None:
        self.log_gamma.grad = None
        self.beta.grad = None

    @torch.no_grad()
    def clamp_(self, max_log_gamma: float, max_beta: float) -> None:
        if max_log_gamma is not None and max_log_gamma > 0:
            self.log_gamma.clamp_(-max_log_gamma, max_log_gamma)
        if max_beta is not None and max_beta > 0:
            self.beta.clamp_(-max_beta, max_beta)

    def magnitude(self) -> Tuple[float, float]:
        return (self.log_gamma.detach().abs().mean().item(),
                self.beta.detach().abs().mean().item())

    def on(self):
        self.active = True

    def off(self):
        self.active = False

    # -- forward ------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        if not self.active:
            return x
        # Lazy device sync. These adapters are intentionally not registered submodules of
        # the owning LightningModule (so they stay out of the head optimizer and out of the
        # checkpoint), which also means `LightningModule.to()` never moves them. The hooks
        # are live during *every* forward -- including Lightning's sanity-check validation,
        # before any explicit sync point could run -- so the check belongs here.
        if self.log_gamma.device != x.device:
            self.to(x.device)
        n = x.size(0)
        if n > self.max_batch:
            raise RuntimeError(
                f'LatentAffineAdapter[{self.swivel}] sized for max_batch={self.max_batch} '
                f'but received a batch of {n}. Increase max_batch.'
            )
        # out-of-place: the fuser/backbone use inplace ReLU, mutating `out` would break autograd
        return torch.exp(self.log_gamma[:n]) * x + self.beta[:n]


LOGITS_SITE = 'logits'


def attach_latent_adapters(
    wrapper: nn.Module,
    sites: Sequence[str],
    max_batch: int,
    input_shape: Sequence[int] = (1, 1, 384, 384),
    device: str = 'cuda:0',
    placement: str = 'post',
    adapt_logits: bool = False,
) -> Tuple[List[LatentAffineAdapter], List[torch.utils.hooks.RemovableHandle]]:
    """Register affine adapters on `wrapper.model` at `sites`.

    adapt_logits=True
        Additionally appends an adapter with `swivel == LOGITS_SITE` acting on the UNet
        output (forward hook on `wrapper.model`), independent of `placement`.  It is always
        the LAST entry of the returned list.

    placement='post' (default)
        Forward hook on the norm's OUTPUT -- the affine survives, so both the segmentation
        and the QC head see it.  These are the reported numbers.

    placement='pre'
        Forward PRE hook on the norm's INPUT, registered with `prepend=True` so it runs
        before the `ScorePredictionWrapper` pre-hook that records the swivel features.
        The head therefore reads adapted features, while the segmentation path sees
        `IN(gamma*x + beta) == IN(x)`, i.e. the affine annihilated (up to eps and
        sign(gamma)).  Use it to measure how much of a mu-hat gain is head-side only.

    Returns a plain list (NOT an nn.ModuleList) on purpose -- see LatentAffineAdapter.
    """
    placement = str(placement).lower()
    if placement not in ('post', 'pre'):
        raise ValueError(f"placement must be 'post' or 'pre', got {placement!r}")

    # adn.N is shape-preserving, so the recorded output shape is also the input shape and
    # the channel count is valid for either placement.
    shapes = find_shapes_for_swivels(wrapper.model, list(sites), tuple(input_shape))
    if adapt_logits:
        # before any adapter hook exists, so this is the plain network output
        with torch.no_grad():
            shapes[LOGITS_SITE] = list(
                wrapper.model(torch.zeros(tuple(input_shape), device=device)).shape
            )
    adapters: List[LatentAffineAdapter] = []
    handles = []
    for site in sites:
        adapter = LatentAffineAdapter(
            swivel=site, channels=shapes[site][1], max_batch=max_batch, device=device,
            placement=placement,
        )
        adapters.append(adapter)
        layer = wrapper.model.get_submodule(site)
        # NOTE `a=adapter` binds now (late binding in a loop is a real bug)
        if placement == 'post':
            # prepend: a post-norm score predictor (hook_placement='post') records the
            # norm output with its own forward hook, registered earlier.  Hooks run in
            # registration order, so without prepend the head would read the UNadapted
            # features at every site that is also one of its swivels.
            handles.append(
                layer.register_forward_hook(
                    lambda m, inp, out, a=adapter: a(out), prepend=True
                )
            )
        else:
            # prepend: must run before ScorePredictionWrapper's recording pre-hook,
            # otherwise the head would read the *un*adapted features and the experiment
            # would silently measure nothing.
            handles.append(
                layer.register_forward_pre_hook(
                    lambda m, inp, a=adapter: a(inp[0]), prepend=True
                )
            )
    if adapt_logits:
        adapter = LatentAffineAdapter(
            swivel=LOGITS_SITE, channels=shapes[LOGITS_SITE][1], max_batch=max_batch,
            device=device, placement='post',
        )
        adapters.append(adapter)
        # hook on the network root: fires after the full forward (all swivel recordings are
        # done), and its return value replaces the logits ScorePredictionWrapper sees.
        handles.append(
            wrapper.model.register_forward_hook(lambda m, inp, out, a=adapter: a(out))
        )
    return adapters, handles


def latent_parameters(adapters: Sequence[LatentAffineAdapter]) -> List[nn.Parameter]:
    return [p for a in adapters for p in a.parameters_list()]


def reset_adapters(adapters: Sequence[LatentAffineAdapter], rows: Optional[slice] = None) -> None:
    for a in adapters:
        a.reset(rows)


# def zero_latent_grads(adapters: Sequence[LatentAffineAdapter]) -> None:
#     for a in adapters:
#         a.zero_grad_()


# def latent_penalty(adapters: Sequence[LatentAffineAdapter], rows: Optional[slice] = None) -> Tensor:
#     """L2 trust-region penalty around the identity."""
#     terms = []
#     for a in adapters:
#         lg = a.log_gamma if rows is None else a.log_gamma[rows]
#         b = a.beta if rows is None else a.beta[rows]
#         terms.append(lg.pow(2).mean() + b.pow(2).mean())
#     return torch.stack(terms).mean()


# --------------------------------------------------------------------------------------
# ascent step
# --------------------------------------------------------------------------------------

# @torch.no_grad()
# def latent_ascent_step(
#     adapters: Sequence[LatentAffineAdapter],
#     rows: slice,
#     gradient_factor: float,
#     max_log_gamma: float = 0.0,
#     max_beta: float = 0.0,
# ) -> float:
#     """One normalized ascent step on mu-hat for the given rows.

#     Mirrors `ScorePredictionWrapperLightningModule.adversarial_step` but
#       * ascends instead of descends (the caller backwards +sum(mu-hat)), and
#       * normalizes per row over the channel axis.

#     Because the update is normalized, it is invariant to any constant scaling of the loss
#     that produced the gradients -- which is what lets us scale the adversarial objective
#     for fp16 headroom without changing step semantics.  Returns the mean raw grad norm
#     (useful to detect fp16 underflow: a value of 0 means no signal reached the adapters).
#     """
#     norms = []
#     for a in adapters:
#         for p in a.parameters_list():
#             if p.grad is None:
#                 norms.append(0.0)
#                 continue
#             g = p.grad[rows]
#             # Drop non-finite gradients rather than propagating them into the cached
#             # gamma/beta, where they would silently NaN out every later forward.
#             g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
#             n = g.flatten(1).norm(2, dim=1).view(-1, 1, 1, 1)
#             norms.append(n.mean().item())
#             p[rows] = p[rows] + gradient_factor * g / (n + 1e-8)
#         a.clamp_(max_log_gamma, max_beta)
#     return float(sum(norms) / max(len(norms), 1))


# --------------------------------------------------------------------------------------
# stopping rules
# --------------------------------------------------------------------------------------

def should_stop(
    rule: str,
    trace: List[Dict[str, float]],
    *,
    max_steps: int,
    delta_mu_target: float = 0.10,
    fg_drift_tol: float = 0.15,
) -> bool:
    """Label-free stopping rules for the TTA loop.

    `trace` is the list of per-step records produced by `run_latent_tta`.

    fixed-steps      : run exactly max_steps (baseline).
    delta-mu-target  : stop once mu-hat has risen by delta_mu_target.
    fg-drift         : stop when the foreground fraction has grown by more than
                       fg_drift_tol (relative) over the un-adapted prediction.
                       mu-hat and kappa-hat rise monotonically even while true Dice
                       collapses, so they carry no stopping information; prediction-mass
                       drift is the most informative label-free signal observed so far.
    """
    if len(trace) <= 1:
        return False
    step = len(trace) - 1
    if step >= max_steps:
        return True
    if rule == 'fixed-steps':
        return False
    if rule == 'delta-mu-target':
        return (trace[-1]['mu'] - trace[0]['mu']) >= delta_mu_target
    if rule == 'fg-drift':
        fg0 = trace[0]['fg_frac']
        if fg0 <= 0:
            return False
        return (trace[-1]['fg_frac'] - fg0) / fg0 >= fg_drift_tol
    raise ValueError(f'unknown stop rule {rule!r}')


# --------------------------------------------------------------------------------------
# inference-time TTA loop
# --------------------------------------------------------------------------------------

def run_latent_tta(
    wrapper: nn.Module,
    adapters: Sequence[LatentAffineAdapter],
    x: Tensor,
    *,
    lr: float = 1e-2,
    max_steps: int = 20,
    stop_rule: str = 'fixed-steps',
    delta_mu_target: float = 0.10,
    fg_drift_tol: float = 0.15,
    prox_weight: float = 0.0,
    max_log_gamma: float = 0.0,
    max_beta: float = 0.0,
    reset: bool = True,
    score_fn: Optional[Callable[[Tensor], Tensor]] = None,
    objective: str = 'mu',
    oracle_fn: Optional[Callable[[Tensor], Tensor]] = None,
    kappa_weight: float = 0.0,
    kappa_objective: str = 'sigma',
    size_weight: float = 0.0,
    size_target: Optional[Tensor] = None,
    fpfn_weight: float = 0.0,
    fpfn_fn: Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]] = None,
    fpfn_form: str = 'separate',
    fpfn_fn_weight: float = 1.0,
    fpfn_source: str = 'oracle',
    hard_fpfn_fn: Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]] = None,
    logit_lr: Optional[float] = None,
) -> Dict:
    """Adapt the latent features of `x` by gradient ascent, then return the result.

    Runs in whatever autograd/precision context the caller provides; the caller is
    expected to use fp32 (the head divides its input by 1000, so d mu-hat / d gamma is
    already ~1e-3 scaled and underflows in fp16).

    Objective
    ---------
    objective='mu'
        Ascend the predicted score mu-hat.  Label-free and deployable.
    objective='oracle-dice'
        Ascend the TRUE (soft) Dice supplied by `oracle_fn(logits) -> [B]`, which must be
        differentiable w.r.t. the logits.  This uses the ground truth and is therefore an
        ORACLE: not deployable, but it bounds what latent adaptation could deliver if the
        quality estimate were perfect, and the gap to objective='mu' is exactly the price
        of an imperfect mu-hat / true-Dice correlation.

    `kappa_weight > 0` additionally descends the QC head's predictive uncertainty at every
    step (simultaneously, in one combined objective, not in an alternating scheme):

        kappa_objective='sigma'      : objective - w * sigma,      sigma = sqrt(mu(1-mu)/(kappa+1))
        kappa_objective='log-kappa'  : objective + w * log(kappa)

    `size_weight > 0` additionally descends the error of the soft foreground size
    S = sum(p) against `size_target` [B, C-1] (fixed for the whole run, per foreground
    class), measured in log space so that it is relative and defined for empty masks:

        objective - w * mean_c (log(1 + S_c) - log(1 + size_target_c))^2

    `fpfn_weight > 0` additionally descends the soft false-positive and false-negative
    counts of the CURRENT mask, recomputed at every step by `fpfn_fn(logits) -> (fp, fn)`
    [B, C-1] (needs labels: ORACLE), anchored at 0 in log space:

        fpfn_form='separate' : objective - w * mean_c (log(1 + FP_c) + log(1 + a * FN_c))
        fpfn_form='joint'    : objective - w * mean_c log(1 + FP_c + a * FN_c)

    with a = `fpfn_fn_weight` (default 1).

    `fpfn_source` selects the counts that enter this term:
        'oracle' : the true soft counts from `fpfn_fn` (labels; default)
        'head'   : the FP/FN head attached to `wrapper` (`wrapper.extra_heads['fpfn']`,
                   see `model.fpfn_head.attach_fpfn_head`); label-free.  `fpfn_fn` may
                   still be given, for reporting only.
    Whenever the head is attached its prediction is recorded (per_step['fp_pred'],
    per_step['fn_pred']), whatever the source.  `hard_fpfn_fn(logits) -> (fp, fn)` [B, C-1]
    gives the true counts of the HARD mask, for reporting only (per_step['fp_hard'],
    per_step['fn_hard']) -- the reference for a head trained on hard counts.

    The separate form is a log-product and is already minimal when either count is 0, so
    it can trade all FP for more FN; the joint form penalises every wrong pixel equally.

    `fpfn_fn` is also evaluated without weight, for reporting only (per_step['fp_soft'],
    per_step['fn_soft']).

    `score_fn(logits) -> [B]` is a *diagnostic* callback for the true (hard) score.  It is
    used for reporting only and must never touch the objective -- with objective='mu' the
    ground truth must not leak into the adaptation.

    Returns a dict with the adapted logits, the predictor output, the per-step trace and
    `per_step`, which holds the SAME quantities at per-sample resolution:

        per_step['mu']         : [B, S]   predicted score after k steps
        per_step['kappa']      : [B, S]   predicted concentration after k steps
        per_step['sigma']      : [B, S]   Beta std implied by (mu, kappa)
        per_step['true_score'] : [B, S]   true hard score after k steps (only if score_fn)
        per_step['soft_dice']  : [B, S]   true soft Dice after k steps (only if oracle_fn)
        per_step['logit_drift']: [B, S]   mean |logits - logits_0|, 0 at step 0
        per_step['soft_size']  : [B, S]   soft foreground size sum(p), summed over classes
        per_step['hard_size']  : [B, S]   hard foreground pixel count, summed over classes
        per_step['size_loss']  : [B, S]   log-size error vs. size_target (only if given)
        per_step['fp_soft'], ['fn_soft'] : [B, S]  true soft FP / FN (only if fpfn_fn)
        per_step['fp_pred'], ['fn_pred'] : [B, S]  FP/FN head prediction (only if attached)
        per_step['fp_hard'], ['fn_hard'] : [B, S]  true hard FP / FN (only if hard_fpfn_fn)

    where S = steps_taken + 1 and column 0 is the un-adapted reference.  The `trace`
    entries are batch means of these, kept scalar because `should_stop` compares them.
    The per-sample matrices are what lets you locate the best step *per image* after the
    fact -- an oracle read-out (it needs labels), for analysis only.

    `logit_drift` is the placement diagnostic: with `placement='pre'` the affine is
    annihilated by the InstanceNorm, so the segmentation cannot move and this stays ~0
    even while mu-hat climbs.

    `logit_lr` sets a separate Adam learning rate for the logit adapter (swivel
    LOGITS_SITE); None uses `lr`.  Adam steps are ~lr in parameter units, and one unit of
    logit beta is a much larger change than one unit of a latent beta.

    If `score_fn` is given, `best_pred` [B,1,H,W] (uint8 labels) holds each sample's hard
    prediction at its best step by true score (first maximum, same as argmax over
    `per_step['true_score']`) -- an oracle read-out for qualitative analysis only.
    """
    objective = str(objective).lower()
    if objective not in ('mu', 'oracle-dice'):
        raise ValueError(f"objective must be 'mu' or 'oracle-dice', got {objective!r}")
    if objective == 'oracle-dice' and oracle_fn is None:
        raise ValueError("objective='oracle-dice' requires a differentiable `oracle_fn`")
    kappa_objective = str(kappa_objective).lower()
    if kappa_weight and kappa_objective not in ('sigma', 'log-kappa'):
        raise ValueError(f"unknown kappa_objective {kappa_objective!r}")
    if size_weight and size_target is None:
        raise ValueError("size_weight > 0 requires a `size_target` [B, C-1]")
    fpfn_form = str(fpfn_form).lower()
    if fpfn_form not in ('separate', 'joint'):
        raise ValueError(f"unknown fpfn_form {fpfn_form!r}")
    fpfn_source = str(fpfn_source).lower()
    if fpfn_source not in ('oracle', 'head'):
        raise ValueError(f"fpfn_source must be 'oracle' or 'head', got {fpfn_source!r}")
    has_head = FPFN_HEAD in getattr(wrapper, 'extra_heads', {})
    if fpfn_weight and fpfn_source == 'oracle' and fpfn_fn is None:
        raise ValueError("fpfn_weight > 0 with fpfn_source='oracle' requires an `fpfn_fn`")
    if fpfn_source == 'head' and not has_head:
        raise ValueError("fpfn_source='head' requires an FP/FN head attached to the wrapper")

    if reset:
        reset_adapters(adapters)
    params = latent_parameters(adapters)
    for p in params:
        p.requires_grad_(True)
    groups = [{'params': a.parameters_list()} for a in adapters if a.swivel != LOGITS_SITE]
    groups += [
        {'params': a.parameters_list(), 'lr': lr if logit_lr is None else logit_lr}
        for a in adapters if a.swivel == LOGITS_SITE
    ]
    opt = torch.optim.Adam(groups, lr=lr)

    trace: List[Dict[str, float]] = []
    # per-sample counterparts of the trace scalars, one [B] tensor per step
    mu_steps: List[Tensor] = []
    kappa_steps: List[Tensor] = []
    sigma_steps: List[Tensor] = []
    score_steps: List[Tensor] = []
    soft_steps: List[Tensor] = []
    drift_steps: List[Tensor] = []
    soft_size_steps: List[Tensor] = []
    hard_size_steps: List[Tensor] = []
    size_loss_steps: List[Tensor] = []
    fp_steps: List[Tensor] = []
    fn_steps: List[Tensor] = []
    fp_pred_steps: List[Tensor] = []
    fn_pred_steps: List[Tensor] = []
    fp_hard_steps: List[Tensor] = []
    fn_hard_steps: List[Tensor] = []
    logits = None
    prediction = None
    logits0 = None
    best_score: Optional[Tensor] = None
    best_pred: Optional[Tensor] = None

    step = 0
    while True:
        with torch.enable_grad():
            logits = wrapper(x)
            prediction = wrapper.prediction
            mu = prediction[:, 0, 0].clamp(1e-6, 1 - 1e-6)  # expected segmentation quality
            kappa = prediction[:, 0, 1].clamp_min(1e-6)     # confidence in mu-hat (inverse variance)
            sigma = (mu * (1 - mu) / (kappa + 1)).sqrt()    # predictive std of the Beta
            # differentiable true Dice -- only entering the objective when objective='oracle-dice',
            # otherwise computed for reporting only (and detached below).
            soft_dice = oracle_fn(logits) if oracle_fn is not None else None
            soft_size = soft_size_from_logits(logits)       # [B, C-1], label-free
            size_loss = (
                (torch.log1p(soft_size) - torch.log1p(size_target.to(soft_size))).pow(2).mean(1)
                if size_target is not None else None
            )
            # true soft FP / FN of the current mask (labels) -- objective only if fpfn_weight > 0
            fp_soft, fn_soft = fpfn_fn(logits) if fpfn_fn is not None else (None, None)
            # FP/FN head prediction (label-free), [B, C-1] counts
            if has_head:
                counts = torch.expm1(wrapper.extra_predictions[FPFN_HEAD].float().clamp_min(0))
                fp_pred, fn_pred = counts[..., 0], counts[..., 1]
            else:
                fp_pred = fn_pred = None
            fp_obj, fn_obj = (fp_pred, fn_pred) if fpfn_source == 'head' else (fp_soft, fn_soft)
            if fp_obj is None:
                fpfn_loss = None
            elif fpfn_form == 'joint':
                fpfn_loss = torch.log1p(fp_obj + fpfn_fn_weight * fn_obj).mean(1)
            else:
                fpfn_loss = (torch.log1p(fp_obj) + torch.log1p(fpfn_fn_weight * fn_obj)).mean(1)

        with torch.no_grad():
            if logits0 is None:
                logits0 = logits.detach().clone()
            binary = logits.shape[1] == 1
            fg = ((logits > 0) if binary else (logits.argmax(1, keepdim=True) > 0))
            drift = (logits - logits0).abs().flatten(1).mean(1).detach().cpu()
            rec = {
                'step': step,
                'mu': mu.mean().item(),
                'kappa': kappa.mean().item(),
                'sigma': sigma.mean().item(),
                'fg_frac': fg.float().mean().item(),
                'logit_drift': float(drift.mean().item()),
            }
            mu_steps.append(mu.detach().cpu())
            kappa_steps.append(kappa.detach().cpu())
            sigma_steps.append(sigma.detach().cpu())
            drift_steps.append(drift)
            soft_size_steps.append(soft_size.sum(1).detach().cpu())
            hard_size_steps.append(fg.flatten(1).sum(1).float().cpu())
            if size_loss is not None:
                size_loss_steps.append(size_loss.detach().cpu())
                rec['size_loss'] = float(size_loss.mean().item())
            if fp_soft is not None:
                fp_steps.append(fp_soft.sum(1).detach().cpu())
                fn_steps.append(fn_soft.sum(1).detach().cpu())
            if fp_pred is not None:
                fp_pred_steps.append(fp_pred.sum(1).detach().cpu())
                fn_pred_steps.append(fn_pred.sum(1).detach().cpu())
            if hard_fpfn_fn is not None:
                fp_h, fn_h = hard_fpfn_fn(logits)
                fp_hard_steps.append(fp_h.sum(1).float().cpu())
                fn_hard_steps.append(fn_h.sum(1).float().cpu())
            if soft_dice is not None:
                sd = soft_dice.detach().cpu()
                soft_steps.append(sd)
                rec['soft_dice'] = float(sd.mean().item())
            if score_fn is not None:
                # keep the per-sample vector; the trace scalar is just its mean
                s = score_fn(logits).detach().cpu()
                score_steps.append(s)
                rec['true_score'] = float(s.mean().item())
                hard = (fg if binary else logits.argmax(1, keepdim=True)).to(torch.uint8)
                if best_score is None:
                    best_score, best_pred = s.clone(), hard.clone()
                else:
                    better = s > best_score  # strict: ties keep the earliest step
                    best_score = torch.where(better, s, best_score)
                    best_pred[better.to(hard.device)] = hard[better.to(hard.device)]
            trace.append(rec)

        if should_stop(stop_rule, trace, max_steps=max_steps,
                       delta_mu_target=delta_mu_target, fg_drift_tol=fg_drift_tol):
            break

        # NOTE:  mu entspricht label-freier Schaetzung des Dice-Scores.
        # Wenn Mu-Hat _genauer_ Schaetzer, dann zeigt Delta-Mu-Hat in Richtung, die den echten Dice verbessert.
        # Die Orakel-Variante ersetzt genau diesen Schaetzer durch den echten (weichen) Dice.
        if objective == 'mu':
            obj = mu.mean()
        else:
            obj = soft_dice.mean()

        # simultaneous descent on the predicted uncertainty (kappa_weight = 0 disables)
        if kappa_weight:
            if kappa_objective == 'sigma':
                obj = obj - kappa_weight * sigma.mean()
            else:  # 'log-kappa'
                obj = obj + kappa_weight * torch.log(kappa).mean()

        # simultaneous descent on the size error vs. the fixed target (size_weight = 0 disables)
        if size_weight:
            obj = obj - size_weight * size_loss.mean()

        # simultaneous descent on the soft FP / FN (true or predicted), anchored at 0 (fpfn_weight = 0 disables)
        if fpfn_weight:
            obj = obj - fpfn_weight * fpfn_loss.mean()

        # currently not in use, but could be re-enabled if we want to keep the adapted parameters close to the identity
        # if prox_weight > 0:
        #     obj = obj - prox_weight * latent_penalty(adapters)
        opt.zero_grad(set_to_none=True)
        (-obj).backward()  # Adam minimizes, so we negate the objective to ascend
        opt.step()
        # currently not in use, but could be re-enabled if we want to keep the adapted parameters close to the identity
        # for a in adapters:
        #     a.clamp_(max_log_gamma, max_beta)
        step += 1

    # final forward with the adapted parameters, no graph
    with torch.no_grad():
        logits = wrapper(x)
        prediction = wrapper.prediction

    # stack over steps -> [B, S], so row i is the full trajectory of sample i
    per_step: Dict[str, Tensor] = {
        'mu': torch.stack(mu_steps, dim=1),
        'kappa': torch.stack(kappa_steps, dim=1),
        'sigma': torch.stack(sigma_steps, dim=1),
        'logit_drift': torch.stack(drift_steps, dim=1),
        'soft_size': torch.stack(soft_size_steps, dim=1),
        'hard_size': torch.stack(hard_size_steps, dim=1),
    }
    if size_loss_steps:
        per_step['size_loss'] = torch.stack(size_loss_steps, dim=1)
    if fp_steps:
        per_step['fp_soft'] = torch.stack(fp_steps, dim=1)
        per_step['fn_soft'] = torch.stack(fn_steps, dim=1)
    if fp_pred_steps:
        per_step['fp_pred'] = torch.stack(fp_pred_steps, dim=1)
        per_step['fn_pred'] = torch.stack(fn_pred_steps, dim=1)
    if fp_hard_steps:
        per_step['fp_hard'] = torch.stack(fp_hard_steps, dim=1)
        per_step['fn_hard'] = torch.stack(fn_hard_steps, dim=1)
    if score_steps:
        per_step['true_score'] = torch.stack(score_steps, dim=1)
    if soft_steps:
        per_step['soft_dice'] = torch.stack(soft_steps, dim=1)

    return {
        'logits': logits.detach(),
        'prediction': prediction.detach(),
        'trace': trace,
        'steps_taken': len(trace) - 1,
        'per_step': per_step,
        'objective': objective,
        'kappa_weight': float(kappa_weight),
        'size_weight': float(size_weight),
        'fpfn_weight': float(fpfn_weight),
        'fpfn_form': fpfn_form,
        'fpfn_fn_weight': float(fpfn_fn_weight),
        'fpfn_source': fpfn_source,
        'best_pred': best_pred,
    }
