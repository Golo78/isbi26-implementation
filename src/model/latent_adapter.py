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
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

import sys
sys.path.append('../')
from utils import find_shapes_for_swivels


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

    def __init__(self, swivel: str, channels: int, max_batch: int, device: str = 'cuda:0'):
        super().__init__()
        self.swivel = swivel
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


def attach_latent_adapters(
    wrapper: nn.Module,
    sites: Sequence[str],
    max_batch: int,
    input_shape: Sequence[int] = (1, 1, 384, 384),
    device: str = 'cuda:0',
) -> Tuple[List[LatentAffineAdapter], List[torch.utils.hooks.RemovableHandle]]:
    """Register post-norm affine adapters on `wrapper.model` at `sites`.

    Returns a plain list (NOT an nn.ModuleList) on purpose -- see LatentAffineAdapter.
    """
    shapes = find_shapes_for_swivels(wrapper.model, list(sites), tuple(input_shape))
    adapters: List[LatentAffineAdapter] = []
    handles = []
    for site in sites:
        adapter = LatentAffineAdapter(
            swivel=site, channels=shapes[site][1], max_batch=max_batch, device=device,
        )
        adapters.append(adapter)
        layer = wrapper.model.get_submodule(site)
        # NOTE post hook, and `a=adapter` to bind now (late binding in a loop is a real bug)
        handles.append(layer.register_forward_hook(lambda m, inp, out, a=adapter: a(out)))
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
) -> Dict:
    """Adapt the latent features of `x` by ascending mu-hat, then return the result.

    Runs in whatever autograd/precision context the caller provides; the caller is
    expected to use fp32 (the head divides its input by 1000, so d mu-hat / d gamma is
    already ~1e-3 scaled and underflows in fp16).

    `score_fn(logits) -> [B]` is an optional *diagnostic* callback for the true score.
    It is used for reporting only and never touches the objective -- keep it that way,
    the ground truth must not leak into the adaptation.

    Returns a dict with the adapted logits, the predictor output, the per-step trace and
    `per_step`, which holds the SAME quantities at per-sample resolution:

        per_step['mu']         : [B, S]   predicted score after k steps
        per_step['true_score'] : [B, S]   true score after k steps (only if score_fn given)

    where S = steps_taken + 1 and column 0 is the un-adapted reference.  The `trace`
    entries are batch means of these, kept scalar because `should_stop` compares them.
    The per-sample matrices are what lets you locate the best step *per image* after the
    fact -- an oracle read-out (it needs labels), for analysis only.
    """
    if reset:
        reset_adapters(adapters)
    params = latent_parameters(adapters)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)

    trace: List[Dict[str, float]] = []
    # per-sample counterparts of the trace scalars, one [B] tensor per step
    mu_steps: List[Tensor] = []
    score_steps: List[Tensor] = []
    logits = None
    prediction = None

    step = 0
    while True:
        with torch.enable_grad():
            logits = wrapper(x)
            prediction = wrapper.prediction
            mu = prediction[:, 0, 0].clamp(1e-6, 1 - 1e-6)  # expected segmentation quality 
            kappa = prediction[:, 0, 1].clamp_min(1e-6)     # confidence in mu-hat (inverse variance)

        with torch.no_grad():
            binary = logits.shape[1] == 1
            fg = ((logits > 0) if binary else (logits.argmax(1, keepdim=True) > 0))
            rec = {
                'step': step,
                'mu': mu.mean().item(),
                'kappa': kappa.mean().item(),
                'sigma': (mu * (1 - mu) / (kappa + 1)).sqrt().mean().item(),
                'fg_frac': fg.float().mean().item(),
            }
            mu_steps.append(mu.detach().cpu())
            if score_fn is not None:
                # keep the per-sample vector; the trace scalar is just its mean
                s = score_fn(logits).detach().cpu()
                score_steps.append(s)
                rec['true_score'] = float(s.mean().item())
            trace.append(rec)

        if should_stop(stop_rule, trace, max_steps=max_steps,
                       delta_mu_target=delta_mu_target, fg_drift_tol=fg_drift_tol):
            break

        # NOTE:  mu entspricht label-freier Schätzung des Dice-Scores. 
        # Wenn Mu-Hat _genauer_ Schätzer, dann zeigt Delta-Mu-Hat in Richtung, die den echten Dice verbessert.
        #  
        objective = mu.mean() 
        # currently not in use, but could be re-enabled if we want to keep the adapted parameters close to the identity
        # if prox_weight > 0:
        #     objective = objective - prox_weight * latent_penalty(adapters)
        opt.zero_grad(set_to_none=True)
        (-objective).backward() # Adam minimizes, so we negate the objective to ascend towards higher mu-hat
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
    per_step: Dict[str, Tensor] = {'mu': torch.stack(mu_steps, dim=1)}
    if score_steps:
        per_step['true_score'] = torch.stack(score_steps, dim=1)

    return {
        'logits': logits.detach(),
        'prediction': prediction.detach(),
        'trace': trace,
        'steps_taken': len(trace) - 1,
        'per_step': per_step,
    }
