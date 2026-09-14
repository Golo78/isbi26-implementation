"""Evaluate latent-space test-time adaptation driven by the confidence predictor.

Keeps the input image fixed and adapts a per-channel affine transform of the latent
features at a subset of the score-predictor swivels, by gradient ascent on the predicted
score mu-hat.  Reports true Dice before and after adaptation per domain, plus the full
per-step trace needed to diagnose whether predicted gains are real gains.

Three switches select the experiment (all default to the original behaviour and are
independent of each other -- run one at a time):

  ++tta.objective=oracle-dice   ORACLE: ascend the TRUE (soft) Dice instead of mu-hat.
                                Uses labels, not deployable; gives the upper bound of what
                                latent adaptation could achieve with a perfect quality
                                estimate, i.e. isolates how much of the gain depends on
                                the mu-hat / true-Dice correlation.
  ++tta.placement=pre           Move the latent adapter from post-norm to PRE-norm.
  ++tta.kappa_weight=0.1        Descend the predictive uncertainty (Beta sigma, or
                                -log kappa via ++tta.kappa_objective=log-kappa)
                                simultaneously with the ascent on the score.

  ++tta.adapt_logits=true       Add a per-channel affine on the UNet logits (in addition to
                                `adapt_sites`; `++tta.adapt_sites=[]` for logits only).

Regardless of the switches, the un-adapted predictions are summarised with their Beta
credible intervals (coverage of the true Dice per nominal level), so the kappa bounds of
the current model can be inspected from the same run.

Every run also computes a THRESHOLD ORACLE on the un-adapted logits: per image, the one
global logit threshold t (binary: `logits > t`; multi-class: per-class logit biases before
argmax) that maximizes the true hard Dice, found by grid search.  It needs labels and is the
reference the "best" TTA results are compared against: anything the adaptation achieves
beyond it is more than recalibrating the final decision threshold.  In addition,
`tta.n_examples` images per domain are snapshotted (image, GT, masks before / at the best
TTA step / after TTA / at the oracle threshold) and plotted by `plot_latent_tta.py` into
`<save_dir>/plots/` together with their predicted-vs-actual Dice and kappa trajectories.
The score predictor's `hook_placement` is taken from the checkpoint metadata.

Written as a standalone app rather than a Lightning `predict_step` so that autograd mode,
precision and batch size are under direct control.

Example
-------
    python eval_latent_tta.py \
        +data=pmri ++data.split=threet-to-onepointfivet \
        +trainer=score_predictor_trainer \
        +unet=monai_unet ++unet.out_channels=1 \
        +model=score_predictor ++model.num_classes=2 \
        ++model.swivels=first-embedding-last \
        +tta=latent_affine \
        ++ckpt_name=cleanHalf_0
"""

import sys, os, glob, pickle as pkl
from typing import Dict, List

import hydra
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append("../")
from data_utils import get_data_module
from model.unet import get_unet_module
from utils import find_unet_checkpoint
from model.score_adapter import get_score_prediction_module
from model.latent_adapter import (
    LOGITS_SITE,
    resolve_swivels,
    attach_latent_adapters,
    reset_adapters,
    # latent_parameters,
    # latent_penalty,
    run_latent_tta,
)
from losses import unified_score_loss, soft_dice_from_logits
from model.calibration import interval_report, format_interval_report
from plot_latent_tta import plot_all


def make_score_fn(target: torch.Tensor, num_classes: int, metric_type: str = "dice"):
    """Diagnostic-only true-score callback. Never used in the adaptation objective."""

    def score_fn(logits: torch.Tensor) -> torch.Tensor:
        preds = (
            (logits > 0).long()
            if logits.shape[1] == 1
            else logits.argmax(1, keepdim=True)
        )
        dummy = torch.zeros(logits.size(0), num_classes, 2, device=logits.device)
        dummy[..., 0] = 0.5
        dummy[..., 1] = 1.0
        _, _, true_score = unified_score_loss(
            predicted_segmentation=preds,
            target_segmentation=target,
            prediction=dummy,
            metric_type=metric_type,
            num_classes=num_classes,
            return_scores=True,
            use_heads="aggregate",
            kappa_reg_weight=0.0,
            beta=0.0,
            likelihood="beta",
        )
        return true_score[:, 0].detach()

    return score_fn


def make_soft_dice_fn(target: torch.Tensor, num_classes: int):
    """Differentiable true-Dice callback (ORACLE objective / diagnostic).

    Unlike `make_score_fn` this keeps the graph, so it can drive the ascent when
    `tta.objective=oracle-dice`.  It reads the ground truth and is therefore only ever an
    upper-bound experiment, never a deployable method.
    """

    def oracle_fn(logits: torch.Tensor) -> torch.Tensor:
        return soft_dice_from_logits(
            logits=logits, target_segmentation=target, num_classes=num_classes
        )

    return oracle_fn


def hard_labels(logits: torch.Tensor) -> torch.Tensor:
    """[B,C,H,W] logits -> [B,H,W] uint8 labels (threshold 0 / argmax)."""
    if logits.shape[1] == 1:
        return (logits[:, 0] > 0).to(torch.uint8)
    return logits.argmax(1).to(torch.uint8)


def threshold_grid(lo: float, hi: float, num: int) -> torch.Tensor:
    """Logit-threshold candidates, always containing 0, ordered by |t|.

    The |t| ordering makes `argmax` resolve ties towards the smallest shift, so an image
    whose Dice is flat around the default threshold reports t = 0 rather than an
    arbitrary grid edge.
    """
    grid = torch.cat([torch.linspace(float(lo), float(hi), int(num)), torch.zeros(1)])
    grid = torch.unique(grid)
    return grid[grid.abs().argsort(stable=True)]


@torch.no_grad()
def threshold_oracle(logits: torch.Tensor, score_fn, shifts: torch.Tensor, sweeps: int = 2):
    """ORACLE: best true hard score reachable by moving only the final decision threshold.

    One threshold per image (not per pixel -- a per-pixel threshold could reproduce the
    ground truth exactly and would bound nothing).  Needs labels via `score_fn`.

    binary  : pred = logits > t, t from `shifts`; exhaustive, returns the full [B, T]
              score matrix so a domain-wide single threshold can be read off as well.
    C > 2   : pred = argmax(logits + b) with a per-class bias b (background fixed at 0),
              coordinate ascent over the foreground classes; `scores` is None.

    t = 0 / b = 0 is always a candidate, so the oracle is never below the un-adapted score.
    """
    if logits.shape[1] == 1:
        scores = torch.stack([score_fn(logits - t) for t in shifts.tolist()], dim=1).cpu()  # [B,T]
        idx = scores.argmax(1)
        best_t = shifts[idx]
        pred = hard_labels(logits - best_t.to(logits.device).view(-1, 1, 1, 1))
        return {
            "dice": scores.gather(1, idx[:, None]).squeeze(1),
            "shift": best_t[:, None],
            "pred": pred,
            "scores": scores,
        }

    B, C = logits.shape[:2]
    bias = torch.zeros(B, C, device=logits.device)
    best = score_fn(logits)
    for _ in range(sweeps):
        for c in range(1, C):
            for t in shifts.tolist():
                cand = bias.clone()
                cand[:, c] = t
                s = score_fn(logits + cand[:, :, None, None])
                better = s > best
                best = torch.where(better, s, best)
                bias[better.to(bias.device)] = cand[better.to(bias.device)]
    return {
        "dice": best,
        "shift": bias.cpu(),
        "pred": hard_labels(logits + bias[:, :, None, None]),
        "scores": None,
    }


@torch.no_grad()
def check_head_reads_adapted(wrapper, adapters, placement, input_shape, device) -> None:
    """Fail fast if the QC head does not see what the latent/logit adapters produce.

    Shifts one adapter at a time by beta=1 and compares what the head records at that
    adapter's site against all adapters at identity.  Where the head reads a site post-norm
    (and the adapter is post-norm) the recorded features must move by exactly +1; the
    head's logits input must move by +1 under the logit adapter.
    """
    x = torch.randn(tuple(input_shape), device=device)

    def head_inputs():
        logits = wrapper(x)
        rec = {a.swivel: a.output[1].detach().clone() for a in wrapper.adapters}
        rec[LOGITS_SITE] = logits.detach().clone()  # the wrapper hands its logits to the head
        return rec

    head_swivels = {a.swivel for a in wrapper.adapters}
    checks = [
        a for a in adapters
        if a.swivel == LOGITS_SITE
        or (placement == "post" and wrapper.placement == "post" and a.swivel in head_swivels)
    ]
    reset_adapters(adapters)
    base = head_inputs()
    for a in checks:
        a.beta[:1] = 1.0
        delta = (head_inputs()[a.swivel] - base[a.swivel]).mean().item()
        a.reset()
        site = a.swivel
        if abs(delta - 1.0) > 1e-3:
            raise RuntimeError(
                f"head input at {site} moved by {delta:.4f} under an adapter shift of 1.0 "
                f"-- the score predictor does not read the adapted features"
            )
    if checks:
        print(f"[info] verified head reads adapted features at: {[a.swivel for a in checks]}")


def concat_per_step(chunks: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-batch [B, S] step matrices into one [N, S_max] matrix.

    Batches only differ in S when an early-stopping rule fires at different points; the
    short ones are padded with NaN so that column k always means "after k ascent steps".
    Use nan-aware reductions on the result.
    """
    if not chunks:
        return torch.empty(0)
    s_max = max(c.shape[1] for c in chunks)
    padded = [
        c
        if c.shape[1] == s_max
        else torch.cat(
            [c, c.new_full((c.shape[0], s_max - c.shape[1]), float("nan"))], dim=1
        )
        for c in chunks
    ]
    return torch.cat(padded, dim=0)


def find_score_predictor_checkpoint(
    base_dir: str, dataset: str, split: str, name: str = "", subdir: str = ""
) -> str:
    # `subdir` names the subfolder of pre-trained/score-predictors to search; the post-norm
    # logits variant is the default.  NOTE it used to be *appended* to that default, so
    # passing ++ckpt_subdir=<default> resolved to <default>/<default> and found nothing.
    folder = subdir or "post-norm_with_logits"
    root = os.path.join(base_dir, folder)
    pattern = os.path.join(
        root,
        f"{dataset}_{split}_*{name}*.ckpt" if name else f"{dataset}_{split}_*.ckpt",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No score predictor checkpoint matching {pattern}")
    if len(matches) > 1:
        print(
            f"[info] {len(matches)} checkpoints matched; using {os.path.basename(matches[0])}"
        )
    return matches[0]


@hydra.main(config_path="../configs", version_base=None)
def main(cfg):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_classes = cfg.model.num_classes
    metric_type = getattr(cfg.model, "loss_fn", "dice")
    cfg_tta = cfg.tta

    print(OmegaConf.to_yaml(cfg))

    # ---- data -----------------------------------------------------------------------
    datamodule = get_data_module(cfg=cfg.data)
    datamodule.setup("test")

    # ---- segmentation network -------------------------------------------------------
    # init model
    # Resolve UNet checkpoint by dataset + split (+optional dropout)
    base_dir = os.path.join(cfg.trainer.root_dir, cfg.unet.checkpoint_dir)
    ckpt_path = find_unet_checkpoint(
        base_dir,
        dataset=cfg.data.dataset,
        split=cfg.data.split,
        dropout=getattr(cfg.unet, "dropout", None),
        newest=True,
    )
    if ckpt_path is None:
        raise FileNotFoundError(
            f"No UNet checkpoint found in {base_dir} for dataset={cfg.data.dataset} split={cfg.data.split}"
        )
    cfg.unet.checkpoint_path = str(ckpt_path)
    unet = get_unet_module(
        cfg=cfg.unet,
        metadata=OmegaConf.to_container(cfg.unet),
        load_from_checkpoint=True,
    ).model

    # ---- confidence predictor -------------------------------------------------------
    sp_ckpt = getattr(cfg, "ckpt_path", None)
    if not sp_ckpt:
        sp_ckpt = find_score_predictor_checkpoint(
            os.path.join(cfg.trainer.root_dir, cfg.trainer.ckpt_dir),
            dataset=cfg.data.dataset,
            split=cfg.data.split,
            name=str(getattr(cfg, "ckpt_name", "") or ""),
            subdir=str(getattr(cfg, "ckpt_subdir", "") or ""),
        )
    print(f"[info] score predictor: {os.path.basename(sp_ckpt)}")

    # The head was trained on either pre- or post-norm swivel features.  Building the
    # wrapper with the other placement loads fine but feeds the head the wrong features,
    # so the checkpoint's own setting is authoritative.
    ckpt_meta = (
        torch.load(str(sp_ckpt), map_location="cpu", weights_only=False)
        .get("hyper_parameters", {})
        .get("metadata")
        or {}
    )
    ckpt_placement = ckpt_meta.get("hook_placement", "pre")
    if str(getattr(cfg.model, "hook_placement", "pre")) != ckpt_placement:
        print(
            f"[warn] model.hook_placement={cfg.model.hook_placement!r} overridden by "
            f"checkpoint metadata: {ckpt_placement!r}"
        )
    cfg.model.hook_placement = ckpt_placement
    print(f"[info] score predictor hook_placement: {ckpt_placement}")

    model = get_score_prediction_module(
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        unet=unet,
        metadata=OmegaConf.to_container(cfg.model),
        ckpt=str(sp_ckpt),
    ).to(device)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # Aktivierungs Gradienten fließen weiter, nur Gewichts-Gradienten werden nicht berechnet
    wrapper = (
        model.wrapper
    )  # ScorePredictionWrapper that returns the logits from the unet

    # head BatchNorm must use running stats and dropout must stay off, otherwise mu-hat
    # depends on the composition of the adaptation batch and the objective is corrupted
    assert not wrapper.training, "wrapper must be in eval mode for TTA"

    # ---- latent adapters ------------------------------------------------------------
    sites = resolve_swivels(
        model=wrapper.model, indices=cfg_tta.adapt_sites
    )  # adapt_sites = [0, 17]

    placement = str(getattr(cfg_tta, "placement", "post"))
    adapt_logits = bool(getattr(cfg_tta, "adapt_logits", False))
    logit_lr = getattr(cfg_tta, "logit_lr", None)
    logit_lr = None if logit_lr is None else float(logit_lr)
    adapters, handles = attach_latent_adapters(
        wrapper,
        sites,
        max_batch=int(cfg.data.batch_size),
        input_shape=OmegaConf.to_object(cfg.data.input_shape),
        device=device,
        placement=placement,
        adapt_logits=adapt_logits,
    )
    logit_adapter = adapters[-1] if adapt_logits else None
    if not adapters:
        raise ValueError("nothing to adapt: tta.adapt_sites is empty and tta.adapt_logits is off")
    check_head_reads_adapted(wrapper, adapters, placement,
                             OmegaConf.to_object(cfg.data.input_shape), device)
    objective = str(getattr(cfg_tta, "objective", "mu"))
    kappa_weight = float(getattr(cfg_tta, "kappa_weight", 0.0))
    kappa_objective = str(getattr(cfg_tta, "kappa_objective", "sigma"))
    thr_cfg = getattr(cfg_tta, "threshold_oracle", None) or {}
    thr_shifts = threshold_grid(
        thr_cfg.get("min", -16.0), thr_cfg.get("max", 16.0), thr_cfg.get("num", 321)
    )
    n_examples = int(getattr(cfg_tta, "n_examples", 6))
    max_batches = getattr(cfg, "max_batches", None)
    print(f"[info] adapting sites: {sites} (placement={placement})")
    if adapt_logits:
        print(f"[info] + logit affine ({LOGITS_SITE}), lr={logit_lr if logit_lr is not None else cfg_tta.lr}")
    print(
        f"[info] objective={objective}"
        + (f" + {kappa_weight} * uncertainty[{kappa_objective}]" if kappa_weight else "")
    )
    if objective == "oracle-dice":
        print("[warn] ORACLE run: the adaptation objective uses ground-truth labels "
              "and is NOT a deployable method -- upper bound only.")

    # ---- evaluate -------------------------------------------------------------------
    results: Dict[str, Dict] = {}
    for domain, loader in datamodule.test_dataloader().items():
        if (
            not getattr(cfg, "include_train", False) and "_train" in domain
        ):  # use only val and test domain
            continue
        print(f"\n=== domain {domain} (n={len(loader.dataset)}) ===")

        n_eval = len(loader.dataset)
        if max_batches is not None:
            n_eval = min(n_eval, int(max_batches) * int(loader.batch_size))
        example_idx = set(
            torch.linspace(0, n_eval - 1, min(n_examples, n_eval)).round().long().tolist()
        ) if n_examples > 0 else set()
        examples: Dict[str, List[torch.Tensor]] = {
            k: []
            for k in ("index", "image", "target", "pred_before", "pred_best",
                      "pred_final", "pred_threshold")
        }
        thr_dice, thr_shift, thr_scores = [], [], []
        logit_log_gamma, logit_beta = [], []
        offset = 0

        rows: Dict[str, List[torch.Tensor]] = {
            k: []
            for k in (
                "dice_before",
                "dice_after",
                "mu_before",
                "mu_after",
                "kappa_before",
                "kappa_after",
                "sigma_before",
                "sigma_after",
                "logit_drift",
            )
        }
        traces, steps_taken = [], []
        # per-image trajectories: one [B, S] matrix per batch, concatenated below
        dice_steps, mu_steps = [], []
        kappa_steps, sigma_steps, soft_dice_steps, drift_steps = [], [], [], []

        for batch_idx, batch in enumerate(tqdm(loader, desc=domain)):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            x = batch["input"].to(device).float()
            t = batch["target"].to(device).long()
            t[t == -1] = 0
            score_fn = make_score_fn(t, num_classes, metric_type)
            # differentiable true Dice: the ascent objective when objective='oracle-dice',
            # a per-step diagnostic otherwise (it never enters the mu objective).
            oracle_fn = make_soft_dice_fn(t, num_classes)

            # un-adapted reference (adapters at identity)
            reset_adapters(adapters)
            with torch.no_grad():
                logits0 = wrapper(x)
                pred0 = wrapper.prediction
            mu0 = pred0[:, 0, 0].detach().cpu()
            kappa0 = pred0[:, 0, 1].detach().cpu()
            rows["dice_before"].append(score_fn(logits0).cpu())
            rows["mu_before"].append(mu0)
            rows["kappa_before"].append(kappa0)
            rows["sigma_before"].append((mu0 * (1 - mu0) / (kappa0 + 1)).sqrt())

            # ORACLE: best Dice by moving only the final threshold of the UN-adapted logits
            thr = threshold_oracle(logits0, score_fn, thr_shifts)
            thr_dice.append(thr["dice"].cpu())
            thr_shift.append(thr["shift"].cpu())
            if thr["scores"] is not None:
                thr_scores.append(thr["scores"].cpu())

            out = run_latent_tta(
                wrapper,
                adapters,
                x,
                lr=cfg_tta.lr,
                max_steps=cfg_tta.max_steps,
                stop_rule=str(cfg_tta.stop_rule),
                delta_mu_target=cfg_tta.delta_mu_target,
                fg_drift_tol=cfg_tta.fg_drift_tol,
                prox_weight=cfg_tta.prox_weight,
                max_log_gamma=cfg_tta.max_log_gamma,
                max_beta=cfg_tta.max_beta,
                reset=True,
                score_fn=score_fn,
                objective=objective,
                oracle_fn=oracle_fn,
                kappa_weight=kappa_weight,
                kappa_objective=kappa_objective,
                logit_lr=logit_lr,
            )
            B = x.size(0)
            if logit_adapter is not None:
                lg, lb = logit_adapter.export_rows(slice(0, B))
                logit_log_gamma.append(lg.flatten(1).cpu())
                logit_beta.append(lb.flatten(1).cpu())

            local = [i - offset for i in sorted(example_idx) if offset <= i < offset + B]
            if local:
                sel = torch.tensor(local, device=x.device)
                examples["index"].append(torch.tensor(local) + offset)
                examples["image"].append(x[sel, 0].half().cpu())
                examples["target"].append(t.view(B, -1, *t.shape[-2:])[sel, 0].to(torch.uint8).cpu())
                examples["pred_before"].append(hard_labels(logits0[sel]).cpu())
                examples["pred_best"].append(out["best_pred"][sel, 0].cpu())
                examples["pred_final"].append(hard_labels(out["logits"][sel]).cpu())
                examples["pred_threshold"].append(thr["pred"][sel].cpu())
            offset += B

            traces.append(out["trace"])
            steps_taken.append(out["steps_taken"])
            dice_steps.append(out["per_step"]["true_score"])
            mu_steps.append(out["per_step"]["mu"])
            kappa_steps.append(out["per_step"]["kappa"])
            sigma_steps.append(out["per_step"]["sigma"])
            drift_steps.append(out["per_step"]["logit_drift"])
            if "soft_dice" in out["per_step"]:
                soft_dice_steps.append(out["per_step"]["soft_dice"])

            mu1 = out["prediction"][:, 0, 0].cpu()
            kappa1 = out["prediction"][:, 0, 1].cpu()
            rows["dice_after"].append(score_fn(out["logits"]).cpu())
            rows["mu_after"].append(mu1)
            rows["kappa_after"].append(kappa1)
            rows["sigma_after"].append((mu1 * (1 - mu1) / (kappa1 + 1)).sqrt())
            rows["logit_drift"].append(out["per_step"]["logit_drift"][:, -1])

        cat = {k: torch.cat(v) for k, v in rows.items()}
        d_before, d_after = (
            cat["dice_before"].mean().item(),
            cat["dice_after"].mean().item(),
        )
        print(
            f"  Dice  {d_before:.4f} -> {d_after:.4f}   (delta {d_after - d_before:+.4f})"
        )
        print(f'  mu    {cat["mu_before"].mean():.4f} -> {cat["mu_after"].mean():.4f}')
        print(
            f'  kappa {cat["kappa_before"].mean():.1f} -> {cat["kappa_after"].mean():.1f}'
            f'   sigma {cat["sigma_before"].mean():.4f} -> {cat["sigma_after"].mean():.4f}'
        )
        # placement diagnostic: a pre-norm affine is annihilated by the InstanceNorm, so
        # the logits (and hence the Dice) cannot move no matter how far mu-hat climbs.
        print(f'  logit drift (mean |logits - logits_0|): {cat["logit_drift"].mean():.6f}')
        if steps_taken:
            print(f"  steps mean {sum(steps_taken)/len(steps_taken):.1f}")

        # kappa credible bounds of the CURRENT (un-adapted) predictions, and after TTA
        bounds_before = interval_report(
            cat["mu_before"], cat["kappa_before"], cat["dice_before"]
        )
        bounds_after = interval_report(
            cat["mu_after"], cat["kappa_after"], cat["dice_after"]
        )
        print(format_interval_report(bounds_before, "  [kappa bounds | before TTA]"))
        print(format_interval_report(bounds_after, "  [kappa bounds | after TTA]"))

        # [N, S]: row = image (same order as dice_before), column k = after k ascent steps
        dice_per_step = concat_per_step(dice_steps)
        mu_per_step = concat_per_step(mu_steps)
        kappa_per_step = concat_per_step(kappa_steps)
        sigma_per_step = concat_per_step(sigma_steps)
        drift_per_step = concat_per_step(drift_steps)
        soft_dice_per_step = (
            concat_per_step(soft_dice_steps) if soft_dice_steps else torch.empty(0)
        )
        assert dice_per_step.shape[0] == cat["dice_before"].numel(), (
            f"per-step matrix has {dice_per_step.shape[0]} rows but "
            f"{cat['dice_before'].numel()} images were evaluated"
        )

        # ORACLE read-out: the best step is picked using the true Dice, so it needs labels
        # and is NOT deployable. It measures what perfect stopping would be worth, i.e. the
        # headroom a label-free rule in `should_stop` is competing for.
        finite = torch.nan_to_num(dice_per_step, nan=float("-inf"))
        best_step = finite.argmax(dim=1)
        best_dice = finite.gather(1, best_step[:, None]).squeeze(1)
        print(
            f"  oracle {best_dice.mean().item():.4f} @ mean step "
            f"{best_step.float().mean().item():.1f} "
            f"(median {int(best_step.median().item())}, "
            f"{int((best_step == dice_per_step.shape[1] - 1).sum())}/{len(best_step)} "
            f"peak at the last step)"
        )
        print(
            f"  per-image delta: {int((cat['dice_after'] > cat['dice_before']).sum())}"
            f"/{len(best_dice)} improved"
        )

        # threshold oracle: per image, and one threshold shared by the whole domain
        thr_dice_all = torch.cat(thr_dice)
        thr_shift_all = torch.cat(thr_shift)
        thr_scores_all = torch.cat(thr_scores) if thr_scores else None
        thr_global_shift, thr_global_dice = None, None
        if thr_scores_all is not None:
            g = int(thr_scores_all.mean(0).argmax())
            thr_global_shift = float(thr_shifts[g])
            thr_global_dice = thr_scores_all[:, g]
        print(
            f"  threshold oracle (per image) {thr_dice_all.mean().item():.4f}"
            + (
                f"   | one threshold for the domain: t={thr_global_shift:+.2f} "
                f"(p={torch.sigmoid(torch.tensor(thr_global_shift)).item():.3f}) "
                f"-> {thr_global_dice.mean().item():.4f}"
                if thr_global_dice is not None else ""
            )
        )
        print(
            f"  best TTA step beats threshold oracle on "
            f"{int((best_dice > thr_dice_all + 1e-6).sum())}/{len(best_dice)} images, "
            f"final TTA on {int((cat['dice_after'] > thr_dice_all + 1e-6).sum())}/{len(best_dice)}"
        )
        if logit_log_gamma:
            lg_all, lb_all = torch.cat(logit_log_gamma), torch.cat(logit_beta)
            msg = f"  logit affine (final): gamma {lg_all.exp().mean():.3f}, beta {lb_all.mean():+.3f}"
            if lg_all.shape[1] == 1:
                implied = -lb_all[:, 0] * torch.exp(-lg_all[:, 0])
                msg += (
                    f", implied threshold {implied.mean():+.3f} "
                    f"(oracle per image {thr_shift_all[:, 0].mean():+.3f})"
                )
            print(msg)

        results[domain] = {
            **{k: v for k, v in cat.items()},
            "delta_dice": d_after - d_before,
            "dice_per_step": dice_per_step,
            "mu_per_step": mu_per_step,
            "kappa_per_step": kappa_per_step,
            "sigma_per_step": sigma_per_step,
            "logit_drift_per_step": drift_per_step,
            "soft_dice_per_step": soft_dice_per_step,
            "best_step": best_step,
            "best_dice": best_dice,
            "traces": traces,
            "steps_taken": steps_taken,
            "sites": sites,
            "objective": objective,
            "placement": placement,
            "kappa_weight": kappa_weight,
            "kappa_objective": kappa_objective,
            "bounds_before": bounds_before,
            "bounds_after": bounds_after,
            # threshold oracle on the un-adapted logits (labels!)
            "thr_grid": thr_shifts,
            "thr_scores": thr_scores_all,          # [N, T] binary only, columns = thr_grid
            "thr_oracle_dice": thr_dice_all,       # [N]
            "thr_oracle_shift": thr_shift_all,     # [N, 1] threshold | [N, C] class biases
            "thr_global_shift": thr_global_shift,  # float | None
            "thr_global_dice": thr_global_dice,    # [N] | None
            "adapt_logits": adapt_logits,
            "logit_log_gamma": torch.cat(logit_log_gamma) if logit_log_gamma else None,  # [N, C]
            "logit_beta": torch.cat(logit_beta) if logit_beta else None,                 # [N, C]
            # snapshots for qualitative comparison; `index` rows into the [N, ...] entries
            "examples": {
                k: (torch.cat(v) if v else torch.empty(0)) for k, v in examples.items()
            },
        }

    for h in handles:
        h.remove()

    # ---- save -----------------------------------------------------------------------
    model_name = os.path.basename(str(sp_ckpt)).replace("0.", "0-").split(".ckpt")[0]
    save_dir = os.path.join(
        cfg.trainer.root_dir,
        "results",
        str(getattr(cfg.trainer, "result_dir", "latent_tta")),
        model_name,
    )
    os.makedirs(save_dir, exist_ok=True)
    packaged = {
        "metadata": {
            "data": OmegaConf.to_container(cfg.data),
            "unet": OmegaConf.to_container(cfg.unet),
            "model": OmegaConf.to_container(cfg.model),
            "tta": OmegaConf.to_container(cfg.tta),
            "score_predictor_ckpt": str(sp_ckpt),
            "unet_ckpt": str(ckpt_path),
        },
        "evaluation": results,
    }
    out_path = os.path.join(save_dir, "latent_tta.pkl")
    with open(out_path, "wb") as f:
        pkl.dump(packaged, f)
    print(f"\nSaved to {out_path}")

    if bool(getattr(cfg_tta, "plot", True)):
        paths = plot_all(packaged, os.path.join(save_dir, "plots"))
        print(f"Wrote {len(paths)} plots to {os.path.join(save_dir, 'plots')}")


if __name__ == "__main__":
    main()
