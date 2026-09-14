"""Plots of a latent TTA run: predicted vs. actual Dice and kappa over the ascent steps, and
the best TTA results next to the threshold oracle.

Reads the `latent_tta.pkl` written by `eval_latent_tta.py` (which calls `plot_all` itself at
the end of every run) and writes, per domain,

  <domain>_summary.png          mean trajectories (actual Dice, mu-hat, kappa), the gain of
                                the best TTA step vs. the threshold oracle per image, and the
                                mean Dice as a function of one domain-wide threshold.
  <domain>_example_<idx>.png    one image: actual Dice vs. mu-hat with its 90% Beta credible
                                band, kappa, and the masks before / at the best TTA step /
                                after TTA / at the oracle threshold.

Overlay colours: green = true positive, red = false positive, blue = false negative.

Usage (from src/apps):
    python plot_latent_tta.py ../../results/<result_dir>/<ckpt>/latent_tta.pkl [--out DIR]
"""

import argparse
import os
import pickle as pkl
import sys
from typing import Dict, List, Optional

import matplotlib

# headless runs: matplotlib otherwise picks the Qt backend here, which aborts/hangs without
# a display.  Notebooks keep their inline backend.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model.calibration import beta_credible_interval


def _row(matrix: torch.Tensor, i: int) -> np.ndarray:
    """Row i of an [N, S] step matrix without the trailing NaN padding."""
    r = matrix[i].float().numpy()
    finite = np.nonzero(~np.isnan(r))[0]
    return r[: finite[-1] + 1] if finite.size else r[:0]


def _nanmean_curve(matrix: torch.Tensor) -> np.ndarray:
    return torch.nanmean(matrix.float(), dim=0).numpy()


def _overlay(img: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Grey image with TP green, FP red, FN blue (foreground = any label > 0)."""
    g = (img - img.min()) / (np.ptp(img) + 1e-8)
    rgb = np.stack([g, g, g], axis=-1)
    gt, pred = gt > 0, pred > 0
    for mask, colour in (
        (gt & pred, (0.1, 0.85, 0.1)),
        (~gt & pred, (0.95, 0.1, 0.1)),
        (gt & ~pred, (0.1, 0.3, 1.0)),
    ):
        rgb[mask] = 0.45 * rgb[mask] + 0.55 * np.asarray(colour)
    return rgb


def _crop_box(masks: List[np.ndarray], shape, margin: int = 24):
    fg = np.zeros(shape, dtype=bool)
    for m in masks:
        fg |= m > 0
    if not fg.any():
        return slice(None), slice(None)
    ys, xs = np.nonzero(fg)
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, shape[0])
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, shape[1])
    return slice(y0, y1), slice(x0, x1)


def plot_example(ev: Dict, k: int, domain: str = "") -> plt.Figure:
    """Figure for the k-th snapshotted example of one domain's evaluation dict."""
    ex = ev["examples"]
    i = int(ex["index"][k])
    dice = _row(ev["dice_per_step"], i)
    mu = _row(ev["mu_per_step"], i)
    kappa = _row(ev["kappa_per_step"], i)
    steps = np.arange(len(dice))
    best = int(ev["best_step"][i])
    thr_dice = float(ev["thr_oracle_dice"][i])
    thr_shift = ev["thr_oracle_shift"][i]
    lo, hi = beta_credible_interval(mu, kappa, coverage=0.9)

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(2, 5, figure=fig, height_ratios=[1, 1.2], hspace=0.3, wspace=0.35)

    ax = fig.add_subplot(gs[0, :3])
    ax.fill_between(steps, lo, hi, color="tab:orange", alpha=0.2, label="90% Beta CI of mu-hat")
    ax.plot(steps, mu, "--", color="tab:orange", lw=2, label="predicted Dice (mu-hat)")
    ax.plot(steps, dice, color="tab:blue", lw=2, label="actual Dice")
    if ev.get("soft_dice_per_step") is not None and ev["soft_dice_per_step"].numel():
        ax.plot(steps, _row(ev["soft_dice_per_step"], i), color="tab:blue", lw=1, alpha=0.4,
                label="actual soft Dice")
    shift_txt = (f"t={float(thr_shift[0]):+.2f}" if thr_shift.numel() == 1
                 else "b=" + ",".join(f"{float(v):+.1f}" for v in thr_shift))
    ax.axhline(thr_dice, color="black", ls=":", lw=1.5,
               label=f"threshold oracle {thr_dice:.3f} ({shift_txt})")
    ax.axvline(best, color="tab:green", lw=1, alpha=0.7, label=f"best TTA step {best}")
    ax.set_xlabel("ascent step")
    ax.set_ylabel("Dice")
    ax.set_title(f"{domain}  image {i}: predicted vs. actual Dice")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    axk = fig.add_subplot(gs[0, 3:])
    axk.plot(steps, kappa, color="tab:purple", lw=2)
    axk.axvline(best, color="tab:green", lw=1, alpha=0.7)
    axk.set_yscale("log")
    axk.set_xlabel("ascent step")
    axk.set_ylabel("kappa")
    axk.set_title("predicted concentration kappa")
    axk.grid(alpha=0.3, which="both")

    img = ex["image"][k].float().numpy()
    gt = ex["target"][k].numpy()
    panels = [
        ("ground truth", gt, None),
        ("before TTA", ex["pred_before"][k].numpy(), dice[0]),
        (f"best TTA step ({best})", ex["pred_best"][k].numpy(), dice[best]),
        (f"after TTA ({len(dice) - 1})", ex["pred_final"][k].numpy(), dice[-1]),
        ("threshold oracle", ex["pred_threshold"][k].numpy(), thr_dice),
    ]
    ys, xs = _crop_box([p[1] for p in panels], gt.shape)
    for col, (title, pred, d) in enumerate(panels):
        a = fig.add_subplot(gs[1, col])
        a.imshow(_overlay(img, gt, pred)[ys, xs], interpolation="nearest")
        a.set_title(title if d is None else f"{title}\nDice {d:.3f}", fontsize=10)
        a.axis("off")
    return fig


def plot_domain_summary(ev: Dict, domain: str = "") -> plt.Figure:
    has_grid = ev.get("thr_scores") is not None
    ncols = 4 if has_grid else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.4))

    ax = axes[0]
    ax.plot(_nanmean_curve(ev["dice_per_step"]), color="tab:blue", lw=2, label="actual Dice")
    ax.plot(_nanmean_curve(ev["mu_per_step"]), "--", color="tab:orange", lw=2,
            label="predicted Dice (mu-hat)")
    ax.axhline(ev["thr_oracle_dice"].mean().item(), color="black", ls=":",
               label="threshold oracle (per image)")
    if ev.get("thr_global_dice") is not None:
        ax.axhline(ev["thr_global_dice"].mean().item(), color="grey", ls="-.",
                   label=f"one threshold for domain (t={ev['thr_global_shift']:+.2f})")
    ax.axhline(ev["best_dice"].mean().item(), color="tab:green", ls="--", lw=1,
               label="best TTA step (per image)")
    ax.set_xlabel("ascent step"); ax.set_ylabel("mean Dice")
    ax.set_title(f"{domain}  (n={len(ev['dice_before'])})")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(_nanmean_curve(ev["kappa_per_step"]), color="tab:purple", lw=2)
    ax.set_yscale("log"); ax.set_xlabel("ascent step"); ax.set_ylabel("mean kappa")
    ax.set_title("predicted concentration kappa"); ax.grid(alpha=0.3, which="both")

    ax = axes[2]
    d0 = ev["dice_before"].numpy()
    gain_thr = ev["thr_oracle_dice"].numpy() - d0
    ax.scatter(gain_thr, ev["best_dice"].numpy() - d0, s=12, alpha=0.6, label="best TTA step")
    ax.scatter(gain_thr, ev["dice_after"].numpy() - d0, s=12, alpha=0.6, label="after TTA")
    lo_, hi_ = ax.get_xlim()[0], max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo_, hi_], [lo_, hi_], "k--", lw=1, label="= threshold oracle")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Dice gain of threshold oracle"); ax.set_ylabel("Dice gain of TTA")
    ax.set_title("per image, relative to un-adapted"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    if has_grid:
        ax = axes[3]
        grid = ev["thr_grid"].numpy()
        order = np.argsort(grid)
        ax.plot(grid[order], ev["thr_scores"].mean(0).numpy()[order], color="black", lw=2)
        ax.axvline(0, color="grey", lw=0.8)
        ax.axvline(ev["thr_global_shift"], color="tab:red", lw=1,
                   label=f"best t={ev['thr_global_shift']:+.2f}")
        ax.set_xlabel("logit threshold t  (pred = logits > t)"); ax.set_ylabel("mean Dice")
        ax.set_title("one threshold for the whole domain"); ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_all(payload: Dict, out_dir: str, domains: Optional[List[str]] = None) -> List[str]:
    """Write summary and example figures for every domain; returns the file paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for domain, ev in payload["evaluation"].items():
        if domains and domain not in domains:
            continue
        if "thr_oracle_dice" not in ev:
            print(f"[skip] {domain}: run predates the threshold oracle / example snapshots")
            continue
        fig = plot_domain_summary(ev, domain)
        paths.append(os.path.join(out_dir, f"{domain}_summary.png"))
        fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
        plt.close(fig)
        for k in range(len(ev["examples"]["index"])):
            fig = plot_example(ev, k, domain)
            idx = int(ev["examples"]["index"][k])
            paths.append(os.path.join(out_dir, f"{domain}_example_{idx:04d}.png"))
            fig.savefig(paths[-1], dpi=110, bbox_inches="tight")
            plt.close(fig)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pkl", help="path to latent_tta.pkl")
    parser.add_argument("--out", default=None, help="output dir (default: <pkl dir>/plots)")
    args = parser.parse_args()
    with open(args.pkl, "rb") as f:
        payload = pkl.load(f)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.pkl)), "plots")
    written = plot_all(payload, out)
    print(f"Wrote {len(written)} plots to {out}")
