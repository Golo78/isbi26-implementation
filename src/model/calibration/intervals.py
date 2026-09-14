"""Credible intervals for the Beta QC head, and how well they cover the true score.

The head predicts (mu, kappa) per image, i.e. a full Beta(mu*kappa, (1-mu)*kappa) over the
segmentation score -- not just a point estimate.  kappa is the concentration: large kappa
means a narrow distribution, i.e. the head claims to be certain about its own mu-hat.

These helpers turn that into the quantity a reviewer actually asks about: an equal-tailed
credible interval [q_{(1-c)/2}, q_{(1+c)/2}] and its *empirical* coverage of the true
Dice.  Coverage well below the nominal level means the head is overconfident (kappa too
large), well above means it is underconfident (kappa too small).

Used by `apps/eval_kappa_bounds.py` (bounds of the current, un-adapted predictions) and by
`apps/eval_latent_tta.py` (bounds before vs. after adaptation).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from .distributions import BetaAdapter


DEFAULT_LEVELS: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95)


def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def beta_credible_interval(
    mu,
    kappa,
    coverage: float = 0.9,
    adapter: Optional[BetaAdapter] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Equal-tailed credible interval of Beta(mu*kappa, (1-mu)*kappa).

    Args:
        mu:       [N] predicted location in (0,1).
        kappa:    [N] predicted concentration > 0.
        coverage: nominal coverage, e.g. 0.9 -> [5%, 95%] quantiles.

    Returns:
        (lo, hi), each [N].
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f'coverage must be in (0,1), got {coverage}')
    adapter = adapter or BetaAdapter()
    mu_np, kappa_np = _np(mu), _np(kappa)
    tail = (1.0 - coverage) / 2.0
    lo = adapter.ppf(np.full_like(mu_np, tail), mu_np, kappa_np)
    hi = adapter.ppf(np.full_like(mu_np, 1.0 - tail), mu_np, kappa_np)
    return np.asarray(lo), np.asarray(hi)


def beta_pit(mu, kappa, y_true, adapter: Optional[BetaAdapter] = None) -> np.ndarray:
    """PIT values F(y_true; mu, kappa).  Uniform on [0,1] iff the head is calibrated."""
    adapter = adapter or BetaAdapter()
    return np.asarray(adapter.cdf(_np(y_true), _np(mu), _np(kappa)))


def interval_report(
    mu,
    kappa,
    y_true,
    levels: Sequence[float] = DEFAULT_LEVELS,
) -> Dict[str, object]:
    """Coverage / width / sharpness diagnostics of the head's credible intervals.

    Returns a dict with, per nominal level, the empirical coverage and mean interval
    width, plus point-estimate quality (MAE, bias, Pearson and Spearman correlation
    between mu-hat and the true score) and the mean PIT.

    The correlation entries are what the ORACLE experiment is compared against: latent TTA
    on mu-hat can only work to the extent that mu-hat ranks/tracks the true Dice.
    """
    adapter = BetaAdapter()
    mu_np, kappa_np, y_np = _np(mu), _np(kappa), _np(y_true)
    finite = np.isfinite(mu_np) & np.isfinite(kappa_np) & np.isfinite(y_np)
    mu_np, kappa_np, y_np = mu_np[finite], kappa_np[finite], y_np[finite]

    per_level = {}
    for c in levels:
        lo, hi = beta_credible_interval(mu_np, kappa_np, coverage=c, adapter=adapter)
        inside = (y_np >= lo) & (y_np <= hi)
        per_level[float(c)] = {
            'nominal': float(c),
            'empirical_coverage': float(inside.mean()) if inside.size else float('nan'),
            'mean_width': float((hi - lo).mean()) if inside.size else float('nan'),
            'median_width': float(np.median(hi - lo)) if inside.size else float('nan'),
            'lo': lo,
            'hi': hi,
            'inside': inside,
            # signed miss: how far outside the interval the truth fell (0 when inside)
            'miss_below': float((np.maximum(lo - y_np, 0)).mean()) if inside.size else float('nan'),
            'miss_above': float((np.maximum(y_np - hi, 0)).mean()) if inside.size else float('nan'),
        }

    pit = beta_pit(mu_np, kappa_np, y_np, adapter=adapter)

    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        if a.size < 2 or a.std() == 0 or b.std() == 0:
            return float('nan')
        return float(np.corrcoef(a, b)[0, 1])

    def _rank(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(a.size, dtype=np.float64)
        return ranks

    return {
        'n': int(mu_np.size),
        'levels': per_level,
        'mu_mean': float(mu_np.mean()) if mu_np.size else float('nan'),
        'kappa_mean': float(kappa_np.mean()) if kappa_np.size else float('nan'),
        'kappa_median': float(np.median(kappa_np)) if kappa_np.size else float('nan'),
        'sigma_mean': float(np.sqrt(mu_np * (1 - mu_np) / (kappa_np + 1)).mean()) if mu_np.size else float('nan'),
        'y_mean': float(y_np.mean()) if y_np.size else float('nan'),
        'mae': float(np.abs(mu_np - y_np).mean()) if mu_np.size else float('nan'),
        'bias': float((mu_np - y_np).mean()) if mu_np.size else float('nan'),
        'pearson': _corr(mu_np, y_np),
        'spearman': _corr(_rank(mu_np), _rank(y_np)),
        'pit_mean': float(pit.mean()) if pit.size else float('nan'),
        'pit': pit,
    }


def format_interval_report(report: Dict[str, object], title: str = '') -> str:
    """One compact block per domain, ready for stdout."""
    lines = []
    if title:
        lines.append(title)
    lines.append(
        f"  n={report['n']}  mu={report['mu_mean']:.4f}  dice={report['y_mean']:.4f}  "
        f"MAE={report['mae']:.4f}  bias={report['bias']:+.4f}  "
        f"rho_p={report['pearson']:.3f}  rho_s={report['spearman']:.3f}"
    )
    lines.append(
        f"  kappa mean={report['kappa_mean']:.1f} median={report['kappa_median']:.1f}  "
        f"sigma={report['sigma_mean']:.4f}  PIT mean={report['pit_mean']:.3f}"
    )
    for c, entry in sorted(report['levels'].items()):
        lines.append(
            f"    {int(round(c * 100)):>3d}% CI: coverage {entry['empirical_coverage'] * 100:5.1f}%  "
            f"width {entry['mean_width']:.4f}  "
            f"(miss below {entry['miss_below']:.4f} / above {entry['miss_above']:.4f})"
        )
    return '\n'.join(lines)
