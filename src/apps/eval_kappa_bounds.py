"""Inspect the kappa credible bounds of the CURRENT (un-adapted) QC predictions.

No adaptation happens here.  For every test domain the score predictor is run once, and
its Beta(mu*kappa, (1-mu)*kappa) posterior over the segmentation score is compared against
the true Dice:

  * equal-tailed credible intervals at several nominal levels, with their EMPIRICAL
    coverage and width -- coverage far below the nominal level means the head is
    overconfident (kappa too large), far above means it is underconfident,
  * the PIT values F(dice_true; mu, kappa), uniform iff the head is calibrated,
  * MAE / bias of mu-hat and its Pearson/Spearman correlation with the true Dice -- the
    quantity the ORACLE experiment (`++tta.objective=oracle-dice`) is contrasted with.

This is the "look at the kappa confidence bounds first" step; the follow-up experiment is
to additionally descend the uncertainty during TTA (`++tta.kappa_weight=...` in
`eval_latent_tta.py`).

Example
-------
    python eval_kappa_bounds.py \
        +data=pmri ++data.split=threet-to-onepointfivet \
        +trainer=score_predictor_trainer \
        +unet=monai_unet ++unet.out_channels=1 \
        +model=score_predictor ++model.num_classes=2 \
        ++model.swivels=first-embedding-last \
        ++ckpt_name=cleanHalf_0
"""

import sys, os, pickle as pkl
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
from model.calibration import interval_report, format_interval_report

# reuse the exact loading / scoring helpers of the TTA app so the two cannot drift apart
from eval_latent_tta import make_score_fn, find_score_predictor_checkpoint


@hydra.main(config_path="../configs", version_base=None)
def main(cfg):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_classes = cfg.model.num_classes
    metric_type = getattr(cfg.model, "loss_fn", "dice")
    levels = tuple(
        OmegaConf.to_object(cfg.levels) if hasattr(cfg, "levels") else (0.5, 0.8, 0.9, 0.95)
    )

    print(OmegaConf.to_yaml(cfg))

    # ---- data -----------------------------------------------------------------------
    datamodule = get_data_module(cfg=cfg.data)
    datamodule.setup("test")

    # ---- segmentation network -------------------------------------------------------
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
    print(f"[info] score predictor: {os.path.basename(str(sp_ckpt))}")

    model = get_score_prediction_module(
        data_cfg=cfg.data,
        model_cfg=cfg.model,
        unet=unet,
        metadata=OmegaConf.to_container(cfg.model),
        ckpt=str(sp_ckpt),
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    wrapper = model.wrapper
    assert not wrapper.training, "wrapper must be in eval mode"

    # ---- evaluate -------------------------------------------------------------------
    results: Dict[str, Dict] = {}
    for domain, loader in datamodule.test_dataloader().items():
        if not getattr(cfg, "include_train", False) and "_train" in domain:
            continue
        print(f"\n=== domain {domain} (n={len(loader.dataset)}) ===")

        rows: Dict[str, List[torch.Tensor]] = {"dice": [], "mu": [], "kappa": []}
        for batch in tqdm(loader, desc=domain):
            x = batch["input"].to(device).float()
            t = batch["target"].to(device).long()
            t[t == -1] = 0
            score_fn = make_score_fn(t, num_classes, metric_type)
            with torch.no_grad():
                logits = wrapper(x)
                pred = wrapper.prediction
            rows["dice"].append(score_fn(logits).cpu())
            rows["mu"].append(pred[:, 0, 0].detach().cpu())
            rows["kappa"].append(pred[:, 0, 1].detach().cpu())

        cat = {k: torch.cat(v) for k, v in rows.items()}
        report = interval_report(cat["mu"], cat["kappa"], cat["dice"], levels=levels)
        print(format_interval_report(report, f"  [kappa bounds] {domain}"))

        results[domain] = {**cat, "bounds": report}

    # ---- save -----------------------------------------------------------------------
    model_name = os.path.basename(str(sp_ckpt)).replace("0.", "0-").split(".ckpt")[0]
    save_dir = os.path.join(
        cfg.trainer.root_dir,
        "results",
        str(getattr(cfg.trainer, "result_dir", "kappa_bounds")),
        model_name,
    )
    os.makedirs(save_dir, exist_ok=True)
    packaged = {
        "metadata": {
            "data": OmegaConf.to_container(cfg.data),
            "unet": OmegaConf.to_container(cfg.unet),
            "model": OmegaConf.to_container(cfg.model),
            "levels": list(levels),
            "score_predictor_ckpt": str(sp_ckpt),
            "unet_ckpt": str(ckpt_path),
        },
        "evaluation": results,
    }
    out_path = os.path.join(save_dir, "kappa_bounds.pkl")
    with open(out_path, "wb") as f:
        pkl.dump(packaged, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
