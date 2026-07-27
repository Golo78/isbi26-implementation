from typing import (
    Callable, 
    Tuple,
    Dict,
    Union,
    List
)
import os
import matplotlib.pyplot as plt
import torch
from torch import (
    nn, 
    Tensor, 
    argmax, 
    stack, 
    flatten, 
    cat,
    corrcoef,
    tensor,
    linspace,
    rand,
    enable_grad
)
from datetime import datetime
from torch.cuda import memory_allocated, empty_cache
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import default_collate
from monai.networks.nets import ResNet, EfficientNet
from monai.networks.layers.factories import Norm

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torchmetrics.functional.classification import (
    dice,
)
from torchvision.transforms import GaussianBlur
import pandas as pd
import seaborn as sns
from copy import deepcopy
from omegaconf import OmegaConf
import wandb
import math
from math import log2
from monai.networks.layers.factories import Conv, Norm
import warnings
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)
import sys

sys.path.append('../')
from utils import find_shapes_for_swivels, eAURC
from losses import dice_per_class_loss, surface_loss, CustomDiceCELoss, unified_score_loss



def custom_init(m):
    if isinstance(m, nn.Conv3d):
        # Use Kaiming normal with mode 'fan_in' to reduce variance
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # Optionally, initialize linear layers with Xavier uniform
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DelayedModelCheckpoint(ModelCheckpoint):
    """Delay monitored checkpoint saving until start_epoch (inclusive)."""
    def __init__(self, start_epoch: int, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = int(start_epoch)
        print(f"[DelayedModelCheckpoint] Active; saving starts at epoch >= {self.start_epoch} (0-based)")

    def _enabled(self, trainer) -> bool:
        return trainer.current_epoch >= self.start_epoch

    def on_validation_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not self._enabled(trainer):
            # light debug every 5 epochs
            if trainer.current_epoch % 5 == 0:
                print(f"[DelayedModelCheckpoint] Block save @ epoch {trainer.current_epoch} (< {self.start_epoch})")
            return
        super().on_validation_epoch_end(trainer, pl_module)

    # Keep these no-ops to avoid accidental saving pathways (save_last not used here)
    def on_train_epoch_end(self, trainer, pl_module):  # type: ignore[override]
        if not self._enabled(trainer):
            return
        super().on_train_epoch_end(trainer, pl_module)

    def on_exception(self, trainer, pl_module, exception):  # type: ignore[override]
        if not self._enabled(trainer):
            print("[DelayedModelCheckpoint] Suppress emergency save before start epoch")
            return
        super().on_exception(trainer, pl_module, exception)



def get_score_prediction_module_trainer(
    data_cfg: OmegaConf,
    model_cfg: OmegaConf,
    trainer_cfg: OmegaConf
):
    # inferred variables
    patience = model_cfg.patience * 2
    now = datetime.now()

    # dataset and split categories (use as-is, no sanitization, and require split)
    dataset_tag = str(getattr(data_cfg, 'dataset'))
    split_tag = getattr(data_cfg, 'split', None)
    if split_tag is None or str(split_tag).strip() == "":
        raise ValueError("data_cfg.split must be provided for naming. No backward compatibility fallback.")
    split_tag = str(split_tag)

    # score tag is the actual loss/score used (no literal 'score' label)
    score_tag = str(getattr(model_cfg, 'loss_fn', 'loss'))

    # perturbation mode for naming: predictor adversarial or none
    perturbation_mode = getattr(model_cfg, 'perturbations', 'none')
    predictor_param = str(getattr(model_cfg, 'predictor_parameter', 'location') or 'location')
    predictor_param_tag = predictor_param.replace('+', '-').replace(',', '-').replace(' ', '')

    if perturbation_mode == 'predictor':
        adv_tag = f"predictor-adversarial-{predictor_param_tag}"
    else:
        adv_tag = 'no-adversarial'
    # collect optional user tag
    user_name = str(getattr(model_cfg, 'name')) if (getattr(model_cfg, 'name', None) and getattr(model_cfg, 'name') != 'None') else ''

    # iteration category: prefer model/run-specific iteration or seed; fallback to 0
    iteration_val = (
        getattr(model_cfg, 'iteration', None)
        if hasattr(model_cfg, 'iteration') else None
    )
    if iteration_val is None:
        iteration_val = getattr(model_cfg, 'seed', None)
    if iteration_val is None:
        iteration_val = getattr(trainer_cfg, 'iteration', None)
    if iteration_val is None:
        iteration_val = 0
    iteration_tag = str(iteration_val)

    # date stamp category
    date_tag = now.strftime("%Y-%m-%d-%H-%M")

    # Filename convention update:
    # - If user provided a name: dataset_split_<score>_<likelihood>-<user_name>_<date>
    # - Else: dataset_split_<score>_<adv>_<likelihood>_<iteration>_<date>
    likelihood_tag = str(getattr(model_cfg, 'likelihood', 'beta'))
    if user_name:
        safe_user = user_name
        filename = f"{dataset_tag}_{split_tag}_{score_tag}_{adv_tag}_{likelihood_tag}-{safe_user}_{date_tag}"
    else:
        filename = f"{dataset_tag}_{split_tag}_{score_tag}_{adv_tag}_{likelihood_tag}_{iteration_tag}_{date_tag}"

    # init logger
    if trainer_cfg.logging:
        wandb.finish()
        logger = WandbLogger(
            project="MIDL-Extension", 
            log_model=True, 
            name=filename
        )
    else:
        logger = None

    # base checkpoint directory (original logic)
    base_ckpt_dir = os.path.join(trainer_cfg.root_dir, trainer_cfg.model_checkpoint.dirpath)
    # ensure base exists (Lightning will also create as needed)
    os.makedirs(base_ckpt_dir, exist_ok=True)

    # Begin checkpointing only after adversarial skip, warmup, AND beta decay phases complete
    start_ckpt_epoch = 150

    # Single directory for all checkpoints (distinguished by filename only)
    val_ckpt = DelayedModelCheckpoint(
        start_epoch=start_ckpt_epoch,
        dirpath=base_ckpt_dir,
        monitor=trainer_cfg.model_checkpoint.monitor,
        mode=trainer_cfg.model_checkpoint.mode,
        save_top_k=trainer_cfg.model_checkpoint.save_top_k,
        filename=filename + '_val-{epoch:02d}-{val_loss:.4f}'
    )

    callbacks = [val_ckpt]

    if getattr(trainer_cfg, 'enable_train_loss_ckpt', True):
        train_ckpt = DelayedModelCheckpoint(
            start_epoch=start_ckpt_epoch,
            dirpath=base_ckpt_dir,
            monitor='train_loss',
            mode='min',
            save_top_k=1,
            filename=filename + '_train-{epoch:02d}-{train_loss:.4f}'
        )
        callbacks.append(train_ckpt)

    ###

    return L.Trainer(
        limit_train_batches=trainer_cfg.limit_train_batches,
        min_epochs=25,
        max_epochs=trainer_cfg.max_epochs,
        logger=logger,
        callbacks=callbacks,
        precision='16-mixed',
        gradient_clip_val=0.5,
        devices=[0],
        inference_mode=False,
        limit_test_batches=50
    )



def get_score_prediction_module(
    data_cfg: OmegaConf,
    model_cfg: OmegaConf,
    unet: nn.Module,
    metadata: Dict,
    ckpt: Union[str, None] = None
):
    ### derived variables
    if model_cfg.swivels in ['best', 'all']:
        swivels = [layer[0] for layer in unet.named_modules()  if 'adn.N' in layer[0]]
    elif model_cfg.swivels == 'ConfidNet':
        swivels = ['model.2.0.adn.A']
    elif model_cfg.swivels == 'embedding-last':
        swivels = [[layer[0] for layer in unet.named_modules()  if 'adn.N' in layer[0]][i] for i in [17, -1]]
    elif model_cfg.swivels == 'embedding':
        swivels = [[layer[0] for layer in unet.named_modules()  if 'adn.N' in layer[0]][i] for i in [17]]
    elif model_cfg.swivels == 'last':
        swivels = [[layer[0] for layer in unet.named_modules()  if 'adn.N' in layer[0]][i] for i in [-1]]
    elif model_cfg.swivels == 'first-embedding-last':
        swivels = [[layer[0] for layer in unet.named_modules()  if 'adn.N' in layer[0]][i] for i in [0, 17, -1]]
    else:
        swivels = [[layer[0] for layer in unet.named_modules()  if 'adn.N' in layer[0]][i] for i in [17, -1]]
        # swivels = [[layer[0] for layer in unet.named_modules()  if 'adn.N' in layer[0]][model_cfg.swivels]]

    output_shapes = find_shapes_for_swivels(
        model=unet, 
        swivels=swivels, 
        input_shape=OmegaConf.to_object(data_cfg.input_shape)
    )

    # Multi-head Beta regression: one aggregate + per-class (foreground classes = num_classes-1)
    # Total heads = num_classes (aggregate + each class), each needs (mu,kappa) => output_dim = 2 * num_classes
    prediction_head = PredictionHead(
        output_shapes=output_shapes,
        output_dim=2 * model_cfg.num_classes
    )

    wrapper = ScorePredictionWrapper(
        model=unet,
        prediction_head=prediction_head,
        adapters=nn.ModuleList([
            ScorePredictionAdapter(
                swivel=swivel,
            ) for swivel in swivels
        ])
    )
    
    wrapper.freeze_normalization_layers()
    wrapper.freeze_model()

    loss_fn_dict = {
        'dice': dice_per_class_loss,
        'surface': surface_loss
    }

    # Create unified loss function with the specified metric type
    def create_unified_loss(metric_type):
        def unified_loss_wrapper(*args, **kwargs):
            return unified_score_loss(*args, metric_type=metric_type, **kwargs)
        unified_loss_wrapper.__name__ = f"{metric_type}_loss"  # Set name for later reference
        return unified_loss_wrapper

    location_target_score_delta_cfg = model_cfg.location_target_score_delta

    if ckpt is None:
        return ScorePredictionWrapperLightningModule(
            wrapper=wrapper,
            loss_fn=model_cfg.loss_fn,
            lr=model_cfg.lr,
            patience=model_cfg.patience,
            location_target_score_delta=location_target_score_delta_cfg,
            num_classes=model_cfg.num_classes,
            perturbations=model_cfg.perturbations,
            adversarial_skip_epochs=getattr(model_cfg, 'adversarial_skip_epochs', 2),
            adversarial_warmup_epochs=getattr(model_cfg, 'adversarial_warmup_epochs', 0),
            beta_decay_epochs=getattr(model_cfg, 'beta_decay_epochs', 0),
            beta_final=getattr(model_cfg, 'beta_final', 0.2),
            likelihood=getattr(model_cfg, 'likelihood', 'beta'),
            predictor_parameter=getattr(model_cfg, 'predictor_parameter', 'location'),
            concentration_target_log_delta=getattr(model_cfg, 'concentration_target_log_delta', math.log(2.0)),
            concentration_controller_gain=getattr(model_cfg, 'concentration_controller_gain', 0.1),
            min_concentration=getattr(model_cfg, 'min_concentration', 1e-6),
            metadata=metadata
        )

    elif isinstance(ckpt, str):
        return ScorePredictionWrapperLightningModule.load_from_checkpoint(
            checkpoint_path=ckpt,
            wrapper=wrapper,
            loss_fn=model_cfg.loss_fn,
            lr=model_cfg.lr,
            patience=model_cfg.patience,
            location_target_score_delta=location_target_score_delta_cfg,
            num_classes=model_cfg.num_classes,
            perturbations=model_cfg.perturbations,
            adversarial_skip_epochs=getattr(model_cfg, 'adversarial_skip_epochs', 2),
            adversarial_warmup_epochs=getattr(model_cfg, 'adversarial_warmup_epochs', 0),
            beta_decay_epochs=getattr(model_cfg, 'beta_decay_epochs', 0),
            beta_final=getattr(model_cfg, 'beta_final', 0.2),
            likelihood=getattr(model_cfg, 'likelihood', 'beta'),
            predictor_parameter=getattr(model_cfg, 'predictor_parameter', 'location'),
            concentration_target_log_delta=getattr(model_cfg, 'concentration_target_log_delta', math.log(2.0)),
            concentration_controller_gain=getattr(model_cfg, 'concentration_controller_gain', 0.1),
            min_concentration=getattr(model_cfg, 'min_concentration', 1e-6),
            metadata=metadata
        )
    


class ResNetBackbone(nn.Module):
    """
    A simple model to predict a score based on a DNN activation.

    Args:
    - input_size (list): The size of the input tensor
    - hidden_dim (int): The size of the hidden layer
    - output_dim (int): The size of the output layer
    - dropout_prob (float): The probability of dropout

    Returns:
    - output (torch.Tensor): The output of the model
    """
    def __init__(
        self, 
        input_size: list, 
        hidden_dim: int = 128, 
        widen_factor=1,
        layers=1,
    ):
        super(ResNetBackbone, self).__init__()

        block='basic'   

        # self.init_bn = nn.BatchNorm2d(input_size[1])
        # self.init_bn.running_var = self.init_bn.running_var * 10000
        self.resnet = ResNet(
            block=block,
            layers=[layers]*4,
            n_input_channels=input_size[1],
            block_inplanes=[input_size[1]] * 4,
            widen_factor=widen_factor,
            spatial_dims=2,
            num_classes=hidden_dim,
            norm=Norm.BATCH
        )

    def forward(self, x):
        # print(x.min(), x.max(), x.mean())
        # x = x.abs()
        # x = x / x.max()
        # x = self.init_bn(x)
        x = x / 1000
        # print(x.min(), x.max(), x.mean())
        return self.resnet(x)
    


class EfficientNetBackbone(nn.Module):
    """
    An EfficientNet-based model to predict a score based on a DNN activation.
    Uses the same interface as ResNetBackbone for consistency.
    Optimized to have similar parameter count as ResNetBackbone.

    Args:
    - input_size (list): The size of the input tensor [batch, channels, height, width]
    - hidden_dim (int): The size of the output feature dimension
    - widen_factor (float): Width multiplier coefficient (w in EfficientNet paper)
    - layers (int): Depth coefficient (affects number of layers through depth_coefficient)

    Returns:
    - output (torch.Tensor): The output of the model
    """
    def __init__(
        self, 
        input_size: list, 
        hidden_dim: int = 128, 
        widen_factor: float = 1.0,
        layers: int = 1,
    ):
        super(EfficientNetBackbone, self).__init__()
        
        # Optimize widen_factor based on layers to match ResNet parameter count
        # For layers=8, use smaller widen_factor; for layers=1, use larger widen_factor
        if layers >= 8:
            optimized_widen_factor = widen_factor * 0.6  # Reduce parameters for deeper networks
        elif layers >= 4:
            optimized_widen_factor = widen_factor * 0.7
        else:
            optimized_widen_factor = widen_factor * 0.75  # Optimal for layers=1
        
        # Use layers parameter to control depth_coefficient
        depth_coefficient = 1.0 + (layers - 1) * 0.025  # Scale depth based on layers
        
        # EfficientNet-B0 block arguments
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        # Calculate image size from input dimensions (assume square for simplicity)
        image_size = max(input_size[2], input_size[3]) if len(input_size) >= 4 else 224
        
        self.efficientnet = EfficientNet(
            blocks_args_str=blocks_args_str,
            spatial_dims=2,
            in_channels=input_size[1],
            num_classes=hidden_dim,  # Output features instead of classification classes
            width_coefficient=optimized_widen_factor,
            depth_coefficient=depth_coefficient,
            dropout_rate=0,
            image_size=image_size,
            norm=Norm.BATCH,
            drop_connect_rate=0,
            depth_divisor=8
        )

    def forward(self, x):
        # Apply same preprocessing as ResNetBackbone
        # x = x / 1000
        return self.efficientnet(x)
    


class PredictionMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x    


class IterativeFeatureFuser(nn.Module):
    def __init__(self, names, in_channels, num_blocks, spatial_dims=2):
        """
        in_channels: {'high':64,'mid':128,'low':256}
        num_blocks:  {'high':2, 'mid':1, 'low':0}
        """
        super().__init__()

        self.names       = [n.replace('.', '-') for n in names]  # avoid '.' in module dict keys
        self.down_blocks = nn.ModuleDict()
        self.fuse_projs  = nn.ModuleDict()

        # normalize keys (replace '.' with '-')
        for key in list(in_channels.keys()):
            in_channels[key.replace('.', '-')] = in_channels.pop(key)
            num_blocks[key.replace('.', '-')]  = num_blocks.pop(key)

        assert len(self.names) == len(in_channels.keys()) == len(num_blocks.keys())

        for src, tgt in zip(self.names, self.names[1:]):
            Ci, Cj = in_channels[src], in_channels[tgt]
            k = num_blocks[src]
            blocks = []
            for _ in range(k):
                blocks.append(
                    Conv[Conv.CONV, spatial_dims](
                        Ci, Cj, kernel_size=3, stride=2, padding=1, bias=False
                    )
                )
                blocks.append(Norm[Norm.BATCH, spatial_dims](Cj))
                blocks.append(nn.ReLU(inplace=True))
                Ci = Cj
            self.down_blocks[src] = (
                nn.Sequential(*blocks) if k > 0 else Conv[Conv.CONV, spatial_dims](
                    Ci, Cj, kernel_size=1, bias=False
                )
            )

            self.fuse_projs[src] = Conv[Conv.CONV, spatial_dims](
                2 * Cj, Cj, kernel_size=1, bias=False
            )

    def forward(self, feats: dict):
        # normalize incoming feature dict keys
        for key in list(feats.keys()):
            feats[key.replace('.', '-')] = feats.pop(key)

        x = feats[self.names[0]]
        for src, tgt in zip(self.names, self.names[1:]):
            x = self.down_blocks[src](x)
            x = torch.cat([x, feats[tgt]], dim=1)
            x = self.fuse_projs[src](x)
        return x


class PredictionHead(nn.Module):
    def __init__(
        self,
        output_shapes,
        hidden_dim: int = 128,
        output_dim: int = 1,
        widen_factor=1,
        layers=8,
    ):
        super().__init__()

        # for layer fuser:
        #   names: from output_shapes, but we need to sort them correctly!
        #   in_channels: from output_shapes[1]
        #   num_blocks: from names order, we do log2(out_t / out_t+1) to get the number of down_sampling blocks
        unordered_names = list(output_shapes.keys())
        # sort names by descending height breaking ties by ascending channels
        names = sorted(
            unordered_names, 
            key=lambda x: (-output_shapes[x][2], output_shapes[x][1])
        )

        in_channels = {key: output_shapes[key][1] for key in output_shapes.keys()}
        num_blocks = {
            names[i]: int(log2(output_shapes[names[i]][2] / output_shapes[names[i+1]][2]))
            for i in range(len(names)-1)
        }
        num_blocks[names[-1]] = 0  # last layer has no downsampling

        self.feature_fuser = IterativeFeatureFuser(
            names=names,
            in_channels=in_channels,
            num_blocks=num_blocks,
            spatial_dims=2
        )

        self.backbone = ResNetBackbone(
            input_size=output_shapes[names[-1]],
            hidden_dim=hidden_dim,
            widen_factor=widen_factor,
            layers=layers
        )

        self.num_heads = output_dim // 2  # each head has (mu,kappa)
        self.fc = PredictionMLP(
            input_dim=hidden_dim,
            output_dim=output_dim  # flattened (mu1,kappa1, mu2,kappa2,...)
        )

        # print(self.feature_fuser, self.backbone, self.fc)

    def forward(self, feats: dict) -> torch.Tensor:
        """
        Forward pass of the prediction head.
        Args:
            feats (dict): Dictionary of features from the backbone.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # fuse features
        fused_feats = self.feature_fuser(feats)
        
        # backbone
        x = self.backbone(fused_feats)

        # print(f'input shape: {x.shape}')

        # raw prediction
        out = self.fc(x)  # [B, 2*H]
        mus = torch.sigmoid(out[:, 0::2])                    # [B,H]
        kappas_raw = torch.nn.functional.softplus(out[:, 1::2])  # [B,H]
        kappas = 1e-3 + (200 - 1e-3) * kappas_raw
        # return stacked heads as [B, H, 2]
        output = torch.stack([mus, kappas], dim=-1)
        return output
        

    
class ScorePredictionAdapter(nn.Module):
    def __init__(
        self,
        swivel: str,
        device: str = 'cuda:0',
    ):
        super().__init__()
        self.swivel = swivel
        self.device = device
        self.active = True

        self.to(device)


    def on(self):
        self.active = True


    def off(self):
        self.active = False



    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        # this adapter only operates when turned on
        if self.active:
            self.output = (self.swivel, x)
        else:
            pass
        return x



class ScorePredictionWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        prediction_head: nn.Module,
        adapters: nn.ModuleList,
        copy: bool = True,
    ):
        super().__init__()
        self.model           = deepcopy(model) if copy else model
        self.prediction_head = prediction_head
        self.adapters        = adapters
        self.adapter_handles = {}
        self.transform       = False
        self.model.eval()

        self.hook_adapters()


    def hook_adapters(
        self,
    ) -> None:
        assert self.adapter_handles == {}, "Adapters already hooked"
        for adapter in self.adapters:
            swivel = adapter.swivel
            layer  = self.model.get_submodule(swivel)
            hook   = self._get_hook(adapter)
            self.adapter_handles[
                swivel
            ] = layer.register_forward_pre_hook(hook)


    def _get_hook(
        self,
        adapter: nn.Module
    ) -> Callable:
        def hook_fn(
            module: nn.Module, 
            x: Tuple[torch.Tensor]
        ) -> torch.Tensor:
            # x, *_ = x # tuple, alternatively use x_in = x[0]
            # x = adapter(x)
            return adapter(x[0])
        
        return hook_fn
    

    def set_transform(self, transform: bool):
        self.transform = transform
        for adapter in self.adapters:
            adapter.transform = transform


    def turn_off_all_adapters(self):
        for adapter in self.adapters:
            adapter.off()

    
    def freeze_model(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False


    def freeze_normalization_layers(self):
        for name, module in self.model.named_modules():
            if 'bn' in name:
                module.eval()


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
    
        logits = self.model(x)
        
        output_per_adapter = {
            adapter.output[0]: adapter.output[1]
            for adapter in self.adapters if adapter.active
        }

        # NOTE: changed last output from output_per_adapter to the logits
        # dimension is the same so works just fine by replacing
        # NOTE: This is hardcoded but should work for now
        output_per_adapter["model.2.0.adn.N"] = logits

        self.prediction = self.prediction_head(
            output_per_adapter
        )

        return logits
    


class ScorePredictionWrapperLightningModule(L.LightningModule):
    def __init__(
        self,
        wrapper: ScorePredictionWrapper,
        loss_fn: Callable,
        lr: float = 1e-6,
        num_classes: int = 4,
        patience: int = 10,
        location_target_score_delta: float = 0.2,
        perturbations: str = 'predictor',
        adversarial_skip_epochs: int = 2,
        adversarial_warmup_epochs: int = 0,
        beta_decay_epochs: int = 0,
        beta_final: float = 0.2,
        likelihood: str = 'beta',
        predictor_parameter: str = 'location',
        concentration_target_log_delta: float = math.log(2.0),
        concentration_controller_gain: float = 0.1,
        min_concentration: float = 1e-6,
        metadata: Dict[str, OmegaConf] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            'wrapper','loss_fn','adversarial_batch','training_step_outputs','val_step_outputs','predict_step_outputs','eaurc'
        ])
        self.wrapper = wrapper
        self.loss_fn = loss_fn
        self.lr = lr
        self.patience = patience
        
        self.location_target_score_delta = float(location_target_score_delta)
        self.num_classes = num_classes
        self.perturbations = perturbations
        self.adversarial_skip_epochs = adversarial_skip_epochs
        self.adversarial_warmup_epochs = adversarial_warmup_epochs
        self.beta_decay_epochs = beta_decay_epochs
        self.beta_final = beta_final
        self.likelihood = likelihood
        self.metadata = metadata
        self.predictor_parameter = (predictor_parameter or 'location').lower()

        raw_tokens = [tok.strip() for tok in self.predictor_parameter.replace('+', ',').split(',') if tok.strip()]
        mode_tokens: List[str] = []
        for tok in raw_tokens:
            if tok == 'both':
                mode_tokens.extend(['location', 'concentration'])
            else:
                mode_tokens.append(tok)
        if not mode_tokens:
            mode_tokens = ['location']
        valid_modes = {'location', 'concentration'}
        invalid = [tok for tok in mode_tokens if tok not in valid_modes]
        if invalid:
            raise ValueError(f"predictor_parameter must be drawn from {valid_modes}, got {invalid}")
        self._predictor_modes = tuple(dict.fromkeys(mode_tokens))
        self.use_location_adversary = 'location' in self._predictor_modes
        self.use_concentration_adversary = 'concentration' in self._predictor_modes
        if (self.use_concentration_adversary or self.use_location_adversary) and self.perturbations not in ['none', 'predictor']:
            raise ValueError(f"perturbations must be one of ['none','predictor'], got {self.perturbations}")

        if self.perturbations not in ['none', 'predictor']:
            raise ValueError(f"perturbations must be one of ['none', 'predictor'], got {self.perturbations}")

        self.model_loss = CustomDiceCELoss(num_classes=num_classes)

        self.adversarial_batch = None
        self.prev_location_score = None
        self.prev_log_concentration = None
        self.prev_true_score = None
        self.step_size_gaps = torch.tensor(0.)

        print("Location target score delta:", self.location_target_score_delta)
        self.location_gradient_factor = 1.0
        self.concentration_gradient_factor = 1.0
        self.target_log_concentration_delta = float(concentration_target_log_delta)
        self.concentration_controller_gain = float(concentration_controller_gain)
        self.min_concentration = float(min_concentration)
        self.training_step_outputs, self.val_step_outputs, self.predict_step_outputs = [], [], []
        self._stored_pre_finetune_best = None

    def current_beta(self) -> float:
        e = self.current_epoch
        phase12_end = self.adversarial_skip_epochs + self.adversarial_warmup_epochs
        if e < phase12_end:
            return 1.0
        decay_start = phase12_end
        decay_end = decay_start + self.beta_decay_epochs
        if self.beta_decay_epochs <= 0 or e >= decay_end:
            return self.beta_final
        t = (e - decay_start) / max(1, self.beta_decay_epochs)
        return 1.0 + (self.beta_final - 1.0) * t

    # Minimal persistence of adaptive adversarial state
    def on_save_checkpoint(self, checkpoint: Dict) -> None:  # type: ignore[override]
        checkpoint['location_gradient_factor'] = float(self.location_gradient_factor)
        checkpoint['concentration_gradient_factor'] = float(self.concentration_gradient_factor)

    def on_load_checkpoint(self, checkpoint: Dict) -> None:  # type: ignore[override]
        if 'location_gradient_factor' in checkpoint:
            self.location_gradient_factor = float(checkpoint['location_gradient_factor'])
        if 'concentration_gradient_factor' in checkpoint:
            self.concentration_gradient_factor = float(checkpoint['concentration_gradient_factor'])



    def forward(self, x):
        return self.wrapper(x)


    def adversarial_step(
        self,
        input: Tensor,
        gradient_factor: float,
        method: str = 'gradient',
    ) -> Tensor:
        assert input.grad.data is not None, "Input tensor requires grad"
        grad = input.grad.data

        if method == 'gradient':   
            gradient_norm = grad.norm(2, dim=((1,2,3))).view(-1, 1, 1, 1)
            adv_input = input.data - gradient_factor * grad / (gradient_norm + 1e-8)

        elif method == 'gradient_sign':
            adv_input = input.data - gradient_factor * grad.sign()

        self.zero_grad()

        return adv_input


    def get_effective_step_factor(self) -> float:
        """Linear warmup factor for adversarial step sizes.

        0 until adversarial_skip_epochs are finished, then linearly ramps from
        0 -> 1 over adversarial_warmup_epochs. If warmup epochs == 0, jumps to 1.
        """
        if self.current_epoch < self.adversarial_skip_epochs:
            return 0.0
        if self.adversarial_warmup_epochs <= 0:
            return 1.0
        ramp_epoch = self.current_epoch - self.adversarial_skip_epochs
        if ramp_epoch >= self.adversarial_warmup_epochs:
            return 1.0
        # +1 so the first ramp epoch is non-zero
        return (ramp_epoch + 1) / self.adversarial_warmup_epochs

    def get_effective_location_target(self) -> float:
        factor = self.get_effective_step_factor()
        return self.location_target_score_delta * factor

    def training_step(self, batch, batch_idx):
        input = batch['input']
        target = batch['target'].long()
        target[target == -1] = 0
        
        use_adversarial = self.perturbations != 'none' and self.current_epoch >= self.adversarial_skip_epochs

        if use_adversarial:
            base_batch = input.size(0)
            metrics: Dict[str, Tensor] = {}
            current_step_size = self.get_effective_location_target()

            # active_predictor = self.perturbations == 'predictor' and (self.use_location_adversary or self.use_concentration_adversary)

            input_chunks = [input]
            target_chunks = [target]
            # print("INPUT SHAPES:", input.shape)
            if self.use_location_adversary:
                if self.adversarial_batch and 'location_input' in self.adversarial_batch:
                    # print(self.adversarial_batch['location_input'].shape)
                    input_chunks.append(self.adversarial_batch['location_input'])
                    target_chunks.append(self.adversarial_batch['target'])
                else:
                    input_chunks.append(input.clone())
                    target_chunks.append(target.clone())

            if self.use_concentration_adversary:
                if self.adversarial_batch and 'concentration_input' in self.adversarial_batch:
                    # print(self.adversarial_batch['concentration_input'].shape)
                    input_chunks.append(self.adversarial_batch['concentration_input'])
                    target_chunks.append(self.adversarial_batch['target'])
                else:
                    input_chunks.append(input.clone())
                    target_chunks.append(target.clone())

            input = cat(input_chunks, dim=0)
            target = cat(target_chunks, dim=0)

            segments: Dict[str, slice] = {}
            start = 0
            segments['clean'] = slice(start, start + base_batch)
            start += base_batch
            # TODO
            if self.use_location_adversary and self.use_concentration_adversary:
                base_batch = base_batch // 2
            if self.use_location_adversary:
                segments['location'] = slice(start, start + base_batch)
                start += base_batch
            if self.use_concentration_adversary:
                segments['concentration'] = slice(start, start + base_batch)
                

            input = input.detach().requires_grad_(True)
            model_logits = self(input)
            if self.num_classes == 2:
                model_prediction = (model_logits > 0).long().detach()
            else:
                model_prediction = argmax(model_logits, dim=1, keepdim=True).detach()

            location_adv_input = None
            concentration_adv_input = None

            location_observed_delta = torch.zeros(1, device=input.device)
            location_delta_step = torch.zeros(1, device=input.device)
            concentration_log_delta = torch.zeros(1, device=input.device)
            concentration_step_error = torch.zeros(1, device=input.device)
            concentration_true_score_delta = torch.zeros(1, device=input.device)

            if self.wrapper.prediction_head:
                predictor_score = self.wrapper.prediction
            else:
                predictor_score = self.wrapper.output_per_adapter

            predictor_true_score_delta_metrics: List[Tensor] = []

            if self.use_location_adversary:
                if self.prev_location_score is not None:
                    loc_slice = segments['location']
                    location_observed_delta = self.prev_location_score - predictor_score[loc_slice, 0, 0].detach()
                location_adv_loss = predictor_score[..., 0].sum()
                location_adv_loss.backward(retain_graph=True)
                location_adv_input = self.adversarial_step(input=input, gradient_factor=self.location_gradient_factor).detach()
                input.grad.zero_()

            if self.use_concentration_adversary:
                if self.prev_log_concentration is not None:
                    conc_slice = segments['concentration']
                    adv_log_kappa = torch.log(predictor_score[conc_slice, 0, 1].detach().clamp_min(self.min_concentration))
                    concentration_log_delta = self.prev_log_concentration - adv_log_kappa
                    concentration_step_error = concentration_log_delta.mean() - self.target_log_concentration_delta
                    self.concentration_gradient_factor -= self.concentration_controller_gain * concentration_step_error.detach().item()
                    self.concentration_gradient_factor = float(max(1e-4, min(20.0, self.concentration_gradient_factor)))
                concentration_adv_loss = -torch.log(predictor_score[..., 1].clamp_min(self.min_concentration)).sum()
                concentration_adv_loss.backward(retain_graph=True)
                concentration_adv_input = self.adversarial_step(input=input, gradient_factor=self.concentration_gradient_factor).detach()
                input.grad.zero_()

            self.adversarial_batch = {'target': target[segments['clean']].detach()}
            #TODO
            if self.use_location_adversary and self.use_concentration_adversary:
                # print(base_batch, location_adv_input.shape)
                self.adversarial_batch['location_input'] = location_adv_input[:base_batch].detach()
                self.adversarial_batch['concentration_input'] = concentration_adv_input[base_batch:2*base_batch].detach()
            else:
                if self.use_location_adversary and location_adv_input is not None:
                    self.adversarial_batch['location_input'] = location_adv_input[segments['clean']].detach()
                if self.use_concentration_adversary and concentration_adv_input is not None:
                    self.adversarial_batch['concentration_input'] = concentration_adv_input[segments['clean']].detach()

            loss, predicted_score, true_score = unified_score_loss(
                predicted_segmentation=model_prediction,
                target_segmentation=target,
                prediction=predictor_score,
                metric_type=self.loss_fn,
                num_classes=self.num_classes,
                return_scores=True,
                use_heads='all',
                kappa_reg_weight=0.001,
                beta=self.current_beta(),
                likelihood=self.likelihood
            )

            if self.prev_true_score is None:
                location_true_score_delta = torch.zeros(1, device=input.device)
                concentration_true_score_delta = torch.zeros(1, device=input.device)
            else:
                #TODO base_batch
                if self.use_location_adversary:
                    loc_slice = segments['location']
                    location_true_score_delta = self.prev_true_score[:base_batch] - true_score[loc_slice, 0].detach()
                else:
                    location_true_score_delta = torch.zeros(1, device=input.device)
                if self.use_concentration_adversary:
                    conc_slice = segments['concentration']
                    concentration_true_score_delta = self.prev_true_score[base_batch:2*base_batch] - true_score[conc_slice, 0].detach()
                else:
                    concentration_true_score_delta = torch.zeros(1, device=input.device)

            if self.prev_true_score is not None and self.use_location_adversary:
                location_delta_step = location_true_score_delta.mean() - current_step_size
                self.location_gradient_factor -= 0.1 * location_delta_step.detach().item()
                self.location_gradient_factor = float(max(1e-4, min(10.0, self.location_gradient_factor)))

            clean_slice = segments['clean']
            reg_slice = slice(clean_slice.stop, loss.size(0)) if loss.size(0) > clean_slice.stop else clean_slice

            metrics.update({
                'loss': loss.detach().mean(0, keepdim=True).cpu(),
                'mse': loss[clean_slice].detach().mean(0, keepdim=True).cpu(),
                'reg': loss[reg_slice].detach().mean(0, keepdim=True).cpu(),
                'true_score': true_score[:, 0:1].detach().cpu(),
                'predicted_score': predicted_score[:, 0:1].detach().cpu(),
            })

            if self.use_location_adversary:
                loc_true_delta = location_true_score_delta.mean().view(-1).detach().cpu()
                predictor_true_score_delta_metrics.append(loc_true_delta)
                metrics["location_observed_delta"] = location_observed_delta.mean().view(-1).detach().cpu()
                metrics["location_step_error"] = location_delta_step.view(-1).detach().cpu()
                metrics["location_gradient_factor"] = tensor(self.location_gradient_factor).view(-1).detach().cpu()
                metrics["location_step_target"] = tensor(current_step_size).view(-1).detach().cpu()

            if self.use_concentration_adversary:
                metrics["concentration_log_delta"] = concentration_log_delta.mean().view(-1).detach().cpu()
                metrics["concentration_step_error"] = concentration_step_error.view(-1).detach().cpu()
                metrics["concentration_gradient_factor"] = tensor(self.concentration_gradient_factor).view(-1).detach().cpu()
                metrics["concentration_step_target"] = tensor(self.target_log_concentration_delta).view(-1).detach().cpu()
                conc_true_delta = concentration_true_score_delta.mean().view(-1).detach().cpu()
                predictor_true_score_delta_metrics.append(conc_true_delta)

            if predictor_true_score_delta_metrics:
                stacked = torch.stack(predictor_true_score_delta_metrics)
                metrics["predictor_true_score_delta"] = stacked.mean(dim=0).view(-1).detach().cpu()


            #TODO
            if self.use_location_adversary and self.use_concentration_adversary:
                self.prev_location_score = predictor_score[:base_batch, 0, 0].detach()
                self.prev_log_concentration = torch.log(predictor_score[base_batch:2*base_batch, 0, 1].detach().clamp_min(self.min_concentration))
            else:

                if self.use_location_adversary:
                    self.prev_location_score = predictor_score[segments['clean'], 0, 0].detach()
                else:
                    self.prev_location_score = None

                if self.use_concentration_adversary:
                    self.prev_log_concentration = torch.log(predictor_score[segments['clean'], 0, 1].detach().clamp_min(self.min_concentration))
                else:
                    self.prev_log_concentration = None

            self.prev_true_score = true_score[segments['clean'], 0].detach()

        else:
            
            logits = self(input)

            if logits.shape[1] == 1:
                preds = (logits > 0).long().detach()
            else:
                preds = argmax(logits, dim=1, keepdim=True).detach()
            if self.wrapper.prediction_head:
                prediction = self.wrapper.prediction
            else:
                prediction = self.wrapper.output_per_adapter
            
            # Use unified loss function without gap regularization
            loss, predicted_score, true_score = unified_score_loss(
                predicted_segmentation=preds, 
                target_segmentation=target,
                prediction=prediction,
                metric_type=self.loss_fn,
                num_classes=self.num_classes,
                return_scores=True,
                use_heads='all',
                kappa_reg_weight=0.001,
                beta=self.current_beta(),
                likelihood=self.likelihood
            )
# 
            # print(f'before: {loss.detach().mean(0, keepdim=True).shape}')
            
            metrics = {
                'loss': loss.detach().mean(0, keepdim=True).cpu(),
                'true_score': true_score[:, 0:1].detach().cpu(),
                'predicted_score': predicted_score[:, 0:1].detach().cpu(),
            }

        self.training_step_outputs.append(metrics)

        return loss.mean()
    

    def on_train_epoch_end(self):
        unregular_metric_keys = [
            'predictor_observed_delta_raw',
        ]

        # merge these two lists and make them unique
        regular_keys = [
            [key for key in self.training_step_outputs[i].keys() if key not in unregular_metric_keys] 
            for i in range(len(self.training_step_outputs))
        ]
        regular_keys = list(set().union(*regular_keys))

        outputs = {
            key: cat([d[key] for d in self.training_step_outputs if key in d], dim=0)
            # key: cat([d[key] for d in self.training_step_outputs], dim=0)
            for key in regular_keys
        }
        self.log('train_loss', outputs['loss'].mean(), on_epoch=True, prog_bar=True, logger=True)


        if self.perturbations != 'none' and self.current_epoch >= self.adversarial_skip_epochs:
            self.log('train_mse', outputs['mse'].mean(), on_epoch=True, logger=True)
            self.log('train_reg', outputs['reg'].mean(), on_epoch=True, logger=True)
            self.log('effective_step_factor', self.get_effective_step_factor(), on_epoch=True, logger=True)
            # Log predictor-specific metrics if predictor perturbations are enabled
            if self.perturbations in ['predictor', 'both']:
                if self.use_location_adversary:
                    if 'location_gradient_factor' in outputs:
                        self.log('location_gradient_factor', outputs['location_gradient_factor'].mean(), on_epoch=True, logger=True)
                    if 'location_step_target' in outputs:
                        self.log('location_step_target', outputs['location_step_target'].mean(), on_epoch=True, logger=True)
                    if 'location_observed_delta' in outputs:
                        self.log('location_observed_delta', outputs['location_observed_delta'].mean(), on_epoch=True, logger=True)
                    if 'location_step_error' in outputs:
                        self.log('location_step_error', outputs['location_step_error'].mean(), on_epoch=True, logger=True)
                if self.use_concentration_adversary:
                    if 'concentration_gradient_factor' in outputs:
                        self.log('concentration_gradient_factor', outputs['concentration_gradient_factor'].mean(), on_epoch=True, logger=True)
                    if 'concentration_step_target' in outputs:
                        self.log('concentration_step_target', outputs['concentration_step_target'].mean(), on_epoch=True, logger=True)
                    if 'concentration_log_delta' in outputs:
                        self.log('concentration_log_delta', outputs['concentration_log_delta'].mean(), on_epoch=True, logger=True)
                    if 'concentration_step_error' in outputs:
                        self.log('concentration_step_error', outputs['concentration_step_error'].mean(), on_epoch=True, logger=True)
                if self.use_location_adversary or self.use_concentration_adversary:
                    if 'predictor_true_score_delta' in outputs:
                        self.log('predictor_true_score_delta', outputs['predictor_true_score_delta'].mean(), on_epoch=True, logger=True)
        else:
            pass

        if 'predictor_observed_delta_raw' in self.training_step_outputs[0].keys():
            predictor_observed_delta_raw = cat(
                [d['predictor_observed_delta_raw'] for d in self.training_step_outputs], dim=0
            )
            # print(f'predictor_observed_delta_raw: {predictor_observed_delta_raw.shape}')
            fig, ax = plt.subplots()
            ax.boxplot(
                predictor_observed_delta_raw.cpu().numpy(),
                vert=False,
                patch_artist=True,
                showmeans=True,         # display the mean
                meanline=True,          # use line instead of triangle
                meanprops={
                    "color": "green",
                    "linewidth": 2,
                    "linestyle": "--",  # dashed green line
                }
            )
            run = self.logger.experiment

            run.log({
                'predictor_observed_delta_raw': wandb.Image(
                    fig,
                    caption=f'Predictor Observed Delta Raw - Epoch {self.current_epoch}'
                )
            })
            plt.close(fig)

        self.training_step_outputs.clear()

    # @torch.enable_grad() 
    # @torch.inference_mode(False)
    @torch.enable_grad()
    def shared_eval_step(
        self, 
        batch, 
        batch_idx, 
        dataloader_idx=None, 
        adv_loss_calculation=False
    ):
        self.eval()
        # with enable_grad():
        #     with torch.inference_mode(False):
        input = batch['input']
        target = batch['target'].long()
        target[target == -1] = 0
        input = input.detach().requires_grad_(True)
        logits = self(input)
        if logits.shape[1] == 1:
            preds = (logits > 0).long().detach()
        else:
            preds = argmax(logits, dim=1, keepdim=True).detach()
        
        if self.wrapper.prediction_head:
            prediction = self.wrapper.prediction
        else:
            prediction = self.wrapper.output_per_adapter
        assert prediction.isnan().sum() == 0, f"Prediction contains NaN values: {prediction.isnan().sum().item()}"

        # Convert multi-head Beta parameters to point estimate (mu) for evaluation
        eps = 1e-6
        mu_eval = prediction[..., 0].clamp(eps, 1 - eps)
        kappa_eval = prediction[..., 1].clamp_min(eps)
        expected_score = mu_eval  # [B,P]

        # Plain NLL (beta=0) for regular input
        regular_plain_nll, reg_predicted_score, reg_true_score = unified_score_loss(
            predicted_segmentation=preds,
            target_segmentation=target,
            prediction=prediction,
            metric_type=self.loss_fn,
            num_classes=self.num_classes,
            return_scores=True,
            use_heads='aggregate',
            kappa_reg_weight=0.0,
            beta=0.0,
            likelihood=self.likelihood
        )
        mse_mu_regular = (reg_true_score[:,0:1] - mu_eval[:,0:1]).pow(2).mean().view(1,).detach().cpu()

        # Base (regular) metrics; adversarial & totals added conditionally
        metrics = {
            'true_score': reg_true_score[:, 0:1].detach().cpu(),
            'predicted_score': expected_score[:, 0:1].detach().cpu(),
            'mu': mu_eval.detach().cpu(),
            'kappa': kappa_eval.detach().cpu(),
            'plain_nll_regular': regular_plain_nll.mean().view(1,).detach().cpu(),
            'mse_mu_regular': mse_mu_regular,
        }

        if adv_loss_calculation:
            location_adv_loss = prediction[..., 0].sum()
            location_adv_loss.backward()
            
            # Compute adversarial prediction with single gradient factor
            gradient_factor = self.location_gradient_factor
            step_size = self.location_target_score_delta
            print(gradient_factor)
            predictor_adv_input = self.adversarial_step(input=input, gradient_factor=gradient_factor).detach()
            logits_adv = self.forward(predictor_adv_input)
            if logits_adv.shape[1] == 1:
                preds_adv = (logits_adv > 0).long().detach()
            else:
                preds_adv = argmax(logits_adv, dim=1, keepdim=True).detach()
            if self.wrapper.prediction_head:
                prediction_adv = self.wrapper.prediction
            else:
                prediction_adv = self.wrapper.output_per_adapter
            
            adv_mu_eval = prediction_adv[..., 0].clamp(eps, 1 - eps)
            adv_kappa_eval = prediction_adv[..., 1].clamp_min(eps)

            adv_plain_nll, adv_predicted_score, adv_true_score = unified_score_loss(
                predicted_segmentation=preds_adv,
                target_segmentation=target,
                prediction=prediction_adv,
                metric_type=self.loss_fn,
                num_classes=self.num_classes,
                return_scores=True,
                use_heads='aggregate',
                kappa_reg_weight=0.0,
                beta=0.0,
                likelihood=self.likelihood,
            )
            adv_mse = (adv_true_score[:,0:1] - prediction_adv[...,0][:,0:1]).pow(2).mean().view(1,)
            step_size_gap = ((reg_predicted_score - adv_predicted_score) - step_size).mean().abs()

            adv_plain_nll_mean = adv_plain_nll.mean().view(1,).detach().cpu()
            adv_mse_mu = adv_mse.detach().cpu()
            step_size_gap_loss = step_size_gap.view(1,).detach().cpu()

            total_plain_nll = regular_plain_nll.mean().detach().cpu() + adv_plain_nll_mean
            total_mse_mu = (mse_mu_regular + adv_mse_mu)
            metrics.update({
                'adv_predicted_score': adv_predicted_score.detach().cpu(),
                'adv_true_score': adv_true_score.detach().cpu(),
                'adv_mu': adv_mu_eval.detach().cpu(),
                'adv_kappa': adv_kappa_eval.detach().cpu(),
                'plain_nll_adv': adv_plain_nll_mean,
                'mse_mu_adv': adv_mse_mu.detach().cpu(),
                'total_plain_nll': total_plain_nll,
                'total_mse_mu': total_mse_mu,
                'step_size_gap': step_size_gap_loss,
                # per-batch provisional loss (final val_loss decided in compute_validation_loss)
                'loss': total_plain_nll,
            })
        else:
            total_plain_nll = regular_plain_nll.mean().detach().cpu()
            metrics.update({
                'total_plain_nll': total_plain_nll,
                'total_mse_mu': mse_mu_regular,  # identical when no adv path
                'loss': total_plain_nll,
            })

        return metrics


    def on_validation_epoch_start(self):
        # Compute step size gap starting the first epoch we will log it
        if self.current_epoch >= (self.adversarial_skip_epochs + 1) and self.perturbations in ['predictor', 'both']:
            effective_size = self.get_effective_location_target()
            self.step_size_gaps = (
                self.trainer.callback_metrics.get('predictor_observed_delta', torch.tensor(0.)) - effective_size
            ).abs().detach().cpu()



    def validation_step(self, batch, batch_idx):
        adv_loss_calculation = True if self.perturbations in ['predictor', 'both'] and self.current_epoch >= self.adversarial_skip_epochs else False
        metrics = self.shared_eval_step(batch, batch_idx, adv_loss_calculation=adv_loss_calculation)
        self.val_step_outputs.append(metrics)
        return metrics



    def on_validation_epoch_end(self):

        # for key in self.val_step_outputs[0].keys():
        #     print(key)
        #     print([d[key] for d in self.val_step_outputs])

           
        outputs = {}
        for key in self.val_step_outputs[0].keys():
            vals = [d[key] for d in self.val_step_outputs if key in d]
            # Promote 0-dim tensors (scalars) to shape (1,) for safe concatenation
            promoted = []
            for v in vals:
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    promoted.append(v.view(1))
                else:
                    promoted.append(v)
            try:
                outputs[key] = cat(promoted, dim=0)
            except Exception:
                # Fallback: flatten all to 1-D then concatenate
                print("weird CHATGPT exception")
                flat = [p.view(-1) for p in promoted]
                outputs[key] = cat(flat, dim=0)
        # mae = (outputs['predicted_score'] - outputs['true_score']).abs().mean(0)
        # Compute and log validation loss (customizable via helper)
        val_loss = self.compute_validation_loss(outputs)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, logger=True)

        # Log structured plain NLL metrics
        if 'plain_nll_regular' in outputs:
            self.log('val_plain_nll_regular', outputs['plain_nll_regular'].mean(), on_epoch=True, logger=True)
        if 'plain_nll_adv' in outputs:
            self.log('val_plain_nll_adv', outputs['plain_nll_adv'].mean(), on_epoch=True, logger=True)
        if 'total_plain_nll' in outputs:
            self.log('val_total_plain_nll', outputs['total_plain_nll'].mean(), on_epoch=True, logger=True, prog_bar=True)

        # Log structured MSE(mu) metrics
        if 'mse_mu_regular' in outputs:
            self.log('val_mse_mu_regular', outputs['mse_mu_regular'].mean(), on_epoch=True, logger=True)
        if 'mse_mu_adv' in outputs:
            self.log('val_mse_mu_adv', outputs['mse_mu_adv'].mean(), on_epoch=True, logger=True)
        if 'total_mse_mu' in outputs:
            self.log('val_total_mse_mu', outputs['total_mse_mu'].mean(), on_epoch=True, logger=True)

        # Optional adversarial auxiliary
        if 'step_size_gap' in outputs:
            self.log('val_step_size_gap', outputs['step_size_gap'].mean(), on_epoch=True, logger=True)

        # Track best total plain NLL (acts like early stopping metric proxy)
        if 'total_plain_nll' in outputs:
            current_total_plain_nll = outputs['total_plain_nll'].mean()
            if self._stored_pre_finetune_best is None or current_total_plain_nll < self._stored_pre_finetune_best:
                self._stored_pre_finetune_best = float(current_total_plain_nll.detach().cpu())
                self.log('best_total_plain_nll', current_total_plain_nll, prog_bar=True, logger=True)

    # (kappa-only finetune removed)

        self.val_step_outputs.clear()

    def compute_validation_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute final validation loss from aggregated outputs.

        Currently: plain_nll_regular (+ optional plain_nll_regular when adversarial & warmed up).
        """
        # base = outputs.get('total_plain_nll', outputs.get('plain_nll_regular')).mean()
        # # If adversarial path present and step_size_gap exists, we can (optionally) add it.
        # if 'step_size_gap' in outputs and self.current_epoch >= (self.adversarial_skip_epochs + 1):
        #     return base + outputs['step_size_gap'].mean()

        base = outputs.get("plain_nll_regular").mean()
        if 'plain_nll_adv' in outputs and self.current_epoch >= (self.adversarial_skip_epochs + 1):
            return base + outputs['plain_nll_adv'].mean()
        return base



    def _save_validation_boxplot(self, outputs):
        """Save boxplot of prediction differences to results/temp directory"""

        if self.current_epoch % 10 == 0:
            try:
                # Create results/temp directory if it doesn't exist
                temp_dir = "../../results/temp"
                os.makedirs(temp_dir, exist_ok=True)
                
                # Calculate differences between predicted and true scores
                differences = (outputs['predicted_score'] - outputs['true_score']).cpu().numpy()
                
                # Create figure and boxplot
                plt.figure(figsize=(10, 6))
                
                # If we have multiple predictors, create separate boxplots for each
                if differences.ndim > 1 and differences.shape[1] > 1:
                    # Multiple predictors - create boxplot for each predictor
                    data_for_plot = []
                    labels = []
                    for i in range(differences.shape[1]):
                        data_for_plot.append(differences[:, i])
                        labels.append(f'Predictor {i}')
                    
                    plt.boxplot(data_for_plot, labels=labels)
                    plt.title(f'Prediction Differences by Predictor - Epoch {self.current_epoch}')
                    plt.ylabel('Predicted Score - True Score')
                    plt.xlabel('Predictor')
                else:
                    # Single predictor or flatten all differences
                    if differences.ndim > 1:
                        differences = differences.flatten()
                    plt.boxplot(differences)
                    plt.title(f'Prediction Differences - Epoch {self.current_epoch}')
                    plt.ylabel('Predicted Score - True Score')
                    plt.xticks([1], ['All Predictors'])
                
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Generate filename with epoch and timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"validation_boxplot_epoch_{self.current_epoch:03d}_{timestamp}.png"
                filepath = os.path.join(temp_dir, filename)
                
                # Save the plot
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Validation boxplot saved to: {filepath}")
                
            except Exception as e:
                print(f"Warning: Failed to save validation boxplot: {e}")
        else:
            print(f"Skipping boxplot saving for epoch {self.current_epoch} (not a multiple of 10)")


    def predict_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx < 2:
            adv_loss_calculation = True
        else:
            adv_loss_calculation = False
        print(f'Predict step - dataloader_idx: {dataloader_idx}, adv_loss_calculation: {adv_loss_calculation}')
        metrics = self.shared_eval_step(batch, batch_idx, dataloader_idx, adv_loss_calculation=adv_loss_calculation)
        print(metrics.keys())
        self.predict_step_outputs.append(metrics)

        return metrics
    

    def configure_optimizers(self):
        optimizer = Adam(self.wrapper.prediction_head.parameters(), lr=self.lr)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=self.patience),
                'monitor': 'val_loss',
                'frequency': 1,
            }
        }



def clean_predictions(
    predictions: Dict, 
    datamodule: L.LightningDataModule, 
    model: L.LightningModule
):
    # Build a robust cleaned predictions dict per test dataloader key.
    # Some collected metric tensors may be 0-d (scalars); promote them to shape (1,)
    # before concatenation. If shapes still mismatch, flatten as fallback.
    predictions_clean = {}
    for dl_key, pred_list in zip(datamodule.test_dataloader().keys(), predictions):
        if len(pred_list) == 0:
            continue
        first = pred_list[0]
        cleaned = {}
        for metric_key in first.keys():
            vals = [p[metric_key] for p in pred_list if metric_key in p]
            promoted = []
            for v in vals:
                if isinstance(v, torch.Tensor) and v.dim() == 0:
                    promoted.append(v.view(1))
                else:
                    promoted.append(v)
            try:
                cleaned[metric_key] = cat(promoted, dim=0)
            except Exception:
                # Fallback: flatten all to 1-D
                cleaned[metric_key] = cat([p.view(-1) for p in promoted], dim=0)
        predictions_clean[dl_key] = cleaned

    predictions_clean['swivels'] = [adapter.swivel for adapter in model.wrapper.adapters]

    # sanity check for data matching
    for key, value in predictions_clean.items():
        if key == 'swivels':
            continue
        dataset_length = datamodule.test_dataloader()[key].dataset.__len__()
        assert value['true_score'].shape[0] == dataset_length, f'{key} dataset length mismatch'

    return predictions_clean



def collect_eval_from_predictions(
    predictions: Dict, 
):
    evaluation = {}

    for key, value in predictions.items():
        if key == 'swivels':
            continue

        # init evaluation dictionary for this dataset 
        eval_dict = {}

        # core metrics to track (regular path)
        eval_dict['corr']            = []
        eval_dict['mae']             = []
        eval_dict['predicted_risks'] = []
        eval_dict['original_risks']  = []
        eval_dict['predictor_idx']   = []
        eval_dict['swivel']          = []

        # Pass through extra eval outputs: mixture CDFs, ZI-Beta, regular/adv tensors
        passthrough_keys = set([
            # regular
            'true_score', 'predicted_score', 'mu', 'kappa', 'pi0',
            'plain_nll_regular', 'mse_mu_regular', 'total_plain_nll', 'total_mse_mu',
            # adversarial
            'adv_true_score', 'adv_predicted_score', 'adv_mu', 'adv_kappa',
            'plain_nll_adv', 'mse_mu_adv', 'step_size_gap',
        ])
        for extra_key, tensor_val in value.items():
            if extra_key.startswith('prob_below_') or extra_key in passthrough_keys:
                eval_dict[extra_key] = tensor_val

        # Helper to safely get first column or flatten
        def _first_col_or_flat(t: torch.Tensor) -> torch.Tensor:
            if isinstance(t, torch.Tensor):
                if t.dim() >= 2 and t.size(-1) >= 1:
                    return t[:, 0]
                return t.view(-1)
            return t

        # Regular risks and summary stats
        true_score = value['true_score']
        pred_score = value['predicted_score']
        original_risks = 1 - (true_score.squeeze(1) if isinstance(true_score, torch.Tensor) and true_score.dim() >= 2 else true_score.view(-1))
        predicted_risk = 1 - _first_col_or_flat(pred_score)
        eval_dict['corr'].append(corrcoef(stack([predicted_risk, original_risks], dim=0))[0, 1])
        eval_dict['mae'].append((predicted_risk - original_risks).abs().mean().item())
        eval_dict['predicted_risks'].append(predicted_risk)
        eval_dict['original_risks'].append(original_risks)
        eval_dict['predictor_idx'].append(0)
        eval_dict['swivel'].append(predictions['swivels'][0])

        # Adversarial risks and optional summary stats (if available)
        if 'adv_true_score' in value and 'adv_predicted_score' in value:
            adv_true = value['adv_true_score']
            adv_pred = value['adv_predicted_score']
            adv_true_flat = adv_true.squeeze(1) if isinstance(adv_true, torch.Tensor) and adv_true.dim() >= 2 else adv_true.view(-1)
            adv_original_risks = 1 - adv_true_flat
            # adv_predicted_score may be [N, 1] or [N]
            adv_pred_first = _first_col_or_flat(adv_pred)
            adv_predicted_risk = 1 - adv_pred_first
            eval_dict['adv_original_risks'] = adv_original_risks
            eval_dict['adv_predicted_risks'] = adv_predicted_risk
            # Simple summaries analogous to corr/mae
            try:
                eval_dict['adv_corr'] = corrcoef(stack([adv_predicted_risk, adv_original_risks], dim=0))[0, 1]
            except Exception:
                pass
            try:
                eval_dict['adv_mae'] = (adv_predicted_risk - adv_original_risks).abs().mean().item()
            except Exception:
                pass
        evaluation[key] = eval_dict

    return evaluation
