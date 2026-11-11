"""Train the FP/FN head (model/fpfn_head.py) on a frozen U-Net and evaluate it.

Training mirrors `train_score_predictor.py` (same data module, U-Net checkpoint lookup and
Lightning trainer incl. checkpoint naming).  After training, the best-val checkpoint is
evaluated on every test loader; per-sample predictions/targets and summary metrics
(MAE and Pearson r in log space, soft Dice implied by the predicted counts) are written to
results/<trainer.result_dir>/<ckpt name>/eval_data.pkl.

    python train_fpfn_head.py +train=True +test=True +data=pmri ++data.split=promise12 \
        +unet=monai_unet ++unet.out_channels=1 +model=fpfn_head ++model.name=fpfn_0 \
        +trainer=score_predictor_trainer ++trainer.model_checkpoint.dirpath=pre-trained/fpfn-heads
"""
import sys, os
import pickle as pkl

import hydra
import torch
from omegaconf import OmegaConf

sys.path.append('../')
from data_utils import get_data_module
from model.unet import get_unet_module
from utils import find_unet_checkpoint
from model.score_adapter import get_score_prediction_module_trainer
from model.fpfn_head import get_fpfn_prediction_module, summary_metrics


@hydra.main(config_path='../configs', version_base=None)
def main(cfg):
    datamodule = get_data_module(cfg=cfg.data)

    base_dir = os.path.join(cfg.trainer.root_dir, cfg.unet.checkpoint_dir)
    unet_ckpt = find_unet_checkpoint(
        base_dir, dataset=cfg.data.dataset, split=cfg.data.split,
        dropout=getattr(cfg.unet, 'dropout', None), newest=True,
    )
    if unet_ckpt is None:
        raise FileNotFoundError(f"No UNet checkpoint in {base_dir} for {cfg.data.dataset}/{cfg.data.split}")
    cfg.unet.checkpoint_path = str(unet_ckpt)
    unet = get_unet_module(cfg=cfg.unet, metadata=OmegaConf.to_container(cfg.unet),
                           load_from_checkpoint=True).model

    print(OmegaConf.to_yaml(cfg))

    # the model config + U-Net checkpoint travel with the head (load_fpfn_head reads them)
    metadata = OmegaConf.to_container(cfg.model)
    metadata['unet_checkpoint'] = str(unet_ckpt)
    metadata['data_split'] = str(cfg.data.split)

    ckpt = None
    if not cfg.train:
        ckpt = os.path.join(cfg.trainer.root_dir, cfg.trainer.ckpt_dir, cfg.trainer.ckpt_name)
        assert os.path.exists(ckpt), f'Checkpoint not found at {ckpt}'
    model = get_fpfn_prediction_module(cfg.data, cfg.model, unet, metadata=metadata, ckpt=ckpt)
    trainer = get_score_prediction_module_trainer(
        data_cfg=cfg.data, model_cfg=cfg.model, trainer_cfg=cfg.trainer
    )

    if cfg.train:
        trainer.fit(model=model, datamodule=datamodule)
        ckpt = getattr(trainer.checkpoint_callback, 'best_model_path', None)
        if not ckpt:
            raise RuntimeError('training produced no checkpoint (trainer.ckpt_start_epoch > max_epochs?)')
        print(f'[info] best checkpoint: {ckpt}')

    if cfg.test:
        prediction = trainer.predict(model=model, datamodule=datamodule, ckpt_path=ckpt)
        loaders = datamodule.test_dataloader()
        if len(loaders) == 1:
            prediction = [prediction]
        evaluation = {}
        for key, batches in zip(loaders.keys(), prediction):
            outs = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
            assert outs['pred'].shape[0] == len(loaders[key].dataset), f'{key}: length mismatch'
            outs['summary'] = summary_metrics(outs)
            evaluation[key] = outs
            print(f'[{key}] ' + '  '.join(f'{k} {v:.3f}' for k, v in outs['summary'].items()))

        model_name = os.path.basename(ckpt).replace('0.', '0-').split('.ckpt')[0]
        save_dir = os.path.join(cfg.trainer.root_dir, 'results', cfg.trainer.result_dir, model_name)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'eval_data.pkl'), 'wb') as f:
            pkl.dump({
                'metadata': {
                    'data': OmegaConf.to_container(cfg.data),
                    'unet': OmegaConf.to_container(cfg.unet),
                    'model': metadata,
                    'trainer': OmegaConf.to_container(cfg.trainer),
                    'checkpoint': ckpt,
                },
                'evaluation': evaluation,
            }, f)
        print(f'Saved to {save_dir}/eval_data.pkl')


if __name__ == '__main__':
    main()
