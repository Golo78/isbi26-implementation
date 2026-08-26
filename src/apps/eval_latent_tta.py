"""Evaluate latent-space test-time adaptation driven by the confidence predictor.

Keeps the input image fixed and adapts a per-channel affine transform of the latent
features at a subset of the score-predictor swivels, by gradient ascent on the predicted
score mu-hat.  Reports true Dice before and after adaptation per domain, plus the full
per-step trace needed to diagnose whether predicted gains are real gains.

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
    resolve_swivels,
    attach_latent_adapters,
    reset_adapters,
    # latent_parameters,
    # latent_penalty,
    run_latent_tta,
)
from losses import unified_score_loss


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
    folder = "score-predictors-using-logits"
    root = (
        os.path.join(base_dir, folder, subdir)
        if subdir
        else os.path.join(base_dir, folder)
    )
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

    adapters, handles = attach_latent_adapters(
        wrapper,
        sites,
        max_batch=int(cfg.data.batch_size),
        input_shape=OmegaConf.to_object(cfg.data.input_shape),
        device=device,
    )
    print(f"[info] adapting sites: {sites} ")

    # ---- evaluate -------------------------------------------------------------------
    results: Dict[str, Dict] = {}
    for domain, loader in datamodule.test_dataloader().items():
        if (
            not getattr(cfg, "include_train", False) and "_train" in domain
        ):  # use only val and test domain
            continue
        print(f"\n=== domain {domain} (n={len(loader.dataset)}) ===")

        rows: Dict[str, List[torch.Tensor]] = {
            k: []
            for k in (
                "dice_before",
                "dice_after",
                "mu_before",
                "mu_after",
                "kappa_before",
                "kappa_after",
            )
        }
        traces, steps_taken = [], []
        # per-image trajectories: one [B, S] matrix per batch, concatenated below
        dice_steps, mu_steps = [], []

        for batch in tqdm(loader, desc=domain):
            x = batch["input"].to(device).float()
            t = batch["target"].to(device).long()
            t[t == -1] = 0
            score_fn = make_score_fn(t, num_classes, metric_type)

            # un-adapted reference (adapters at identity)
            reset_adapters(adapters)
            with torch.no_grad():
                logits0 = wrapper(x)
                pred0 = wrapper.prediction
            rows["dice_before"].append(score_fn(logits0).cpu())
            rows["mu_before"].append(pred0[:, 0, 0].detach().cpu())
            rows["kappa_before"].append(pred0[:, 0, 1].detach().cpu())

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
            )
            traces.append(out["trace"])
            steps_taken.append(out["steps_taken"])
            dice_steps.append(out["per_step"]["true_score"])
            mu_steps.append(out["per_step"]["mu"])

            rows["dice_after"].append(score_fn(out["logits"]).cpu())
            rows["mu_after"].append(out["prediction"][:, 0, 0].cpu())
            rows["kappa_after"].append(out["prediction"][:, 0, 1].cpu())

        cat = {k: torch.cat(v) for k, v in rows.items()}
        d_before, d_after = (
            cat["dice_before"].mean().item(),
            cat["dice_after"].mean().item(),
        )
        print(
            f"  Dice  {d_before:.4f} -> {d_after:.4f}   (delta {d_after - d_before:+.4f})"
        )
        print(f'  mu    {cat["mu_before"].mean():.4f} -> {cat["mu_after"].mean():.4f}')
        if steps_taken:
            print(f"  steps mean {sum(steps_taken)/len(steps_taken):.1f}")

        # [N, S]: row = image (same order as dice_before), column k = after k ascent steps
        dice_per_step = concat_per_step(dice_steps)
        mu_per_step = concat_per_step(mu_steps)
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

        results[domain] = {
            **{k: v for k, v in cat.items()},
            "delta_dice": d_after - d_before,
            "dice_per_step": dice_per_step,
            "mu_per_step": mu_per_step,
            "best_step": best_step,
            "best_dice": best_dice,
            "traces": traces,
            "steps_taken": steps_taken,
            "sites": sites,
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


if __name__ == "__main__":
    main()
