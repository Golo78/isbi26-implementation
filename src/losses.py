from typing import Callable, Optional
from torch import (
    Tensor,
    nn,
    isnan,
    where,
    tensor,
    exp,
    cat,
    sigmoid,
    log
)
import torch
from torch.nn.functional import one_hot,binary_cross_entropy_with_logits
from torchvision.ops import sigmoid_focal_loss
from monai.metrics import (
    DiceMetric, 
    compute_hausdorff_distance,
    SurfaceDiceMetric
)
from monai.losses import DiceLoss
from torch.distributions import Beta



def dice_per_class_loss(
    predicted_segmentation: Tensor, 
    target_segmentation: Tensor,
    prediction: Tensor,
    criterion: Callable = nn.MSELoss(reduction='none'),
    num_classes: int = 4,
    return_scores: bool = False
) -> Tensor:
    """
    Calculate the Dice coefficient per class loss.

    Args:
        predicted_segmentation (Tensor): The predicted segmentation tensor.
        target_segmentation (Tensor): The target segmentation tensor.
        prediction (Tensor): The prediction tensor.
        criterion (Callable, optional): The loss function to use. Defaults to nn.MSELoss(reduction='none').
        num_classes (int, optional): The number of classes. Defaults to 4.
        return_scores (bool, optional): Whether to return the prediction and target tensor. Defaults to False.

    Returns:
        Tensor: The calculated loss. If return_scores is True, also returns the score tensor.
    """
    score = DiceMetric(
        include_background=True, 
        reduction="none",
        num_classes=num_classes,
        ignore_empty=False
    )(predicted_segmentation, target_segmentation).detach()
    
    not_nans = ~isnan(score) * 1.0
    not_nans = not_nans.unsqueeze(1).repeat(1, prediction.shape[1], 1)
    score = score.nan_to_num(0).detach().unsqueeze(1).repeat(1, prediction.shape[1], 1).clamp(0, 512)
    
    loss = criterion(prediction, score) * not_nans
    
    assert loss.isnan().sum() == 0, f"Loss contains NaN values: {loss.isnan().sum().item()}"

    if return_scores:
        nan_mask   = where(not_nans == 0, tensor(float('nan')), not_nans)
        prediction = prediction.detach() * nan_mask
        score      = score * nan_mask
        prediction = prediction[..., 1:].nanmean(-1)
        score      = score[..., 1:].nanmean(-1)
        return loss, prediction.nan_to_num(0), score.nan_to_num(0)
    else:
        return loss

    

def soft_dice_from_logits(
    logits: Tensor,
    target_segmentation: Tensor,
    num_classes: int = 4,
    eps: float = 1e-6,
    include_background: bool = False,
) -> Tensor:
    """Differentiable per-sample foreground Dice computed directly from logits.

    The hard Dice used for reporting (`DiceMetric` on argmax/threshold labels) has zero
    gradient almost everywhere, so it cannot drive gradient ascent.  This soft relaxation
    replaces the one-hot prediction by the predicted class probabilities and is the
    objective used by the ORACLE variant of the latent TTA (see `run_latent_tta`,
    objective='oracle-dice'): it needs the ground truth and is therefore NOT deployable --
    it measures the upper bound a perfect quality estimator could reach.

    Agrees with the reported hard Dice in the limit of confident predictions and matches
    its empty-empty convention (`ignore_empty=False`): both are 1.0 when prediction and
    target are empty.

    Args:
        logits:              [B, 1, H, W] (binary, sigmoid) or [B, C, H, W] (softmax).
        target_segmentation: [B, 1, H, W] or [B, H, W] integer labels.
        num_classes:         number of classes including background.
        include_background:  average over all classes instead of foreground only.

    Returns:
        Tensor: [B] soft Dice in (0, 1], differentiable w.r.t. `logits`.
    """
    if logits.shape[1] == 1:
        fg = sigmoid(logits)
        probs = cat([1.0 - fg, fg], dim=1)          # [B, 2, H, W]
    else:
        probs = torch.softmax(logits, dim=1)        # [B, C, H, W]

    n_ch = probs.shape[1]
    assert n_ch == num_classes, \
        f'soft_dice_from_logits: {n_ch} predicted channels but num_classes={num_classes}'

    tgt = target_segmentation
    if tgt.dim() == probs.dim():
        tgt = tgt.squeeze(1)
    tgt_oh = one_hot(tgt.long().clamp_min(0), num_classes=n_ch)
    tgt_oh = tgt_oh.moveaxis(-1, 1).to(probs.dtype)  # [B, C, H, W]

    dims = tuple(range(2, probs.dim()))
    intersection = (probs * tgt_oh).sum(dims)        # [B, C]
    denominator = probs.sum(dims) + tgt_oh.sum(dims) # [B, C]
    dice = (2.0 * intersection + eps) / (denominator + eps)

    return dice.mean(1) if include_background else dice[:, 1:].mean(1)


def hausdorff_loss(
    predicted_segmentation: Tensor, 
    target_segmentation: Tensor,
    prediction: Tensor,
    criterion: Callable = nn.MSELoss(reduction='none'),
    num_classes: int = None,
    sigma: float = 1.0,
    return_scores: bool = False
) -> Tensor:
    """
    Calculate the transformed Hausdorff distance per class loss.

    Args:
        predicted_segmentation (Tensor): The predicted segmentation tensor.
        target_segmentation (Tensor): The target segmentation tensor.
        prediction (Tensor): The prediction tensor.
        criterion (Callable, optional): The loss function to use. Defaults to nn.MSELoss(reduction='none').
        num_classes (int, optional): The number of classes. Defaults to 4.
        sigma (Float, optional): Sigma for RBF transformation. Defaults to 1.0.
        return_scores (bool, optional): Whether to return the prediction and target tensor. Defaults to False.

    Returns:
        Tensor: The calculated loss. If return_scores is True, also returns the score tensor.
    """
    predicted_segmentation = one_hot(predicted_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
    target_segmentation = one_hot(target_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)

    score = compute_hausdorff_distance(
        y_pred=predicted_segmentation,
        y=target_segmentation,
        include_background=True,
        percentile=95,
    ).detach()

    not_nans = ~isnan(score) * 1.0
    not_nans = not_nans.unsqueeze(1).repeat(1, prediction.shape[1], 1).detach()
    score    = score.nan_to_num(0).unsqueeze(1).repeat(1, prediction.shape[1], 1).clamp(0, 512)
    score    = exp(-(score ** 2) / (2 * sigma**2)).detach()
    loss     = criterion(prediction, score) * not_nans
    assert loss.isnan().sum() == 0, f"Loss contains NaN values: {loss.isnan().sum().item()}"

    if return_scores:
        nan_mask   = where(not_nans == 0, tensor(float('nan')), not_nans)
        prediction = prediction.detach() * nan_mask
        score      = score * nan_mask
        prediction = prediction[..., 1:].nanmean(-1)
        score      = score[..., 1:].nanmean(-1)
        return loss, prediction.nan_to_num(0), score.nan_to_num(0)
    else:
        return loss
    


def surface_loss(
    predicted_segmentation: Tensor, 
    target_segmentation: Tensor,
    prediction: Tensor,
    criterion: Callable = nn.MSELoss(reduction='none'),
    num_classes: int = None,
    return_scores: bool = False
) -> Tensor:
    """
    Calculate the transformed Hausdorff distance per class loss.

    Args:
        predicted_segmentation (Tensor): The predicted segmentation tensor.
        target_segmentation (Tensor): The target segmentation tensor.
        prediction (Tensor): The prediction tensor.
        criterion (Callable, optional): The loss function to use. Defaults to nn.MSELoss(reduction='none').
        num_classes (int, optional): The number of classes. Defaults to 4.
        sigma (Float, optional): Sigma for RBF transformation. Defaults to 1.0.
        return_scores (bool, optional): Whether to return the prediction and target tensor. Defaults to False.

    Returns:
        Tensor: The calculated loss. If return_scores is True, also returns the score tensor.
    """
    predicted_segmentation = one_hot(predicted_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
    target_segmentation = one_hot(target_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)

    score = SurfaceDiceMetric(
        include_background=True, 
        reduction="none",
        class_thresholds=[3] * num_classes,
    )(predicted_segmentation, target_segmentation).detach()

    not_nans = ~isnan(score) * 1.0
    not_nans = not_nans.unsqueeze(1).repeat(1, prediction.shape[1], 1).detach()
    score    = score.nan_to_num(0).unsqueeze(1).repeat(1, prediction.shape[1], 1).clamp(0, 512)
    loss     = criterion(prediction, score) * not_nans
    assert loss.isnan().sum() == 0, f"Loss contains NaN values: {loss.isnan().sum().item()}"

    if return_scores:
        nan_mask   = where(not_nans == 0, tensor(float('nan')), not_nans)
        prediction = prediction.detach() * nan_mask
        score      = score * nan_mask
        prediction = prediction[..., 1:].nanmean(-1)
        score      = score[..., 1:].nanmean(-1)
        return loss, prediction.nan_to_num(0), score.nan_to_num(0)
    else:
        return loss



class CustomDiceCELoss(nn.Module):
    def __init__(self, num_classes: int):
        """
        Combines MONAI DiceLoss + CE/BCE, returning per-sample loss.
        Args:
            num_classes: number of output channels.
        """
        super().__init__()
        self.num_classes = num_classes

        # Dice part: if binary, use sigmoid; if multiclass, use softmax+one-hot
        if num_classes == 2:
            # Binary Dice (single output channel)
            self.dice = DiceLoss(
                softmax=False,
                sigmoid=True,
                to_onehot_y=False,
                reduction="none",
                batch=False,
            )  # output shape [B]
            # BCE for class 1 vs background
            self.bce = nn.BCEWithLogitsLoss(reduction="none")
        else:
            # Multiclass Dice (expects logits over C channels)
            self.dice = DiceLoss(
                softmax=True,
                sigmoid=False,
                to_onehot_y=True,
                reduction="none",
                batch=False,
            )  # output shape [B, C]
            self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            logits: Tensor of shape [B, C, ...] where C=num_classes if multiclass,
                    or [B, 1, ...] if binary (num_classes==2).
            target: For multiclass: Tensor [B, ...] with integer labels in [0, C-1].
                    For binary: Tensor [B, ...] with {0,1} or floats.
        Returns:
            combined_loss: Tensor shape [B], each element = Dice+CE (or BCE).
        """
        B = logits.shape[0]

        if self.num_classes == 2:
            # ===== Binary Case =====
            # Dice component: expects (B,1,...) logits and (B,1,...) target
            # MONAI’s DiceLoss for binary returns [B] if batch=False, reduction='none'
            dice = self.dice(logits, target.float())
            dice_per_sample = dice.view(B, -1).mean(dim=1)
            # BCE component: returns [B, *spatial]
            bce_map = self.bce(logits, target.float())
            # Flatten spatial dims and sum per-sample
            bce_per_sample = bce_map.view(B, -1).mean(dim=1)

            # print(bce_per_sample.shape, dice_per_sample.shape)
            # Combine (weights = 1 each)
            return dice_per_sample + bce_per_sample

        else:
            # ===== Multiclass Case =====
            # Dice component: logits [B,C,...], target [B,...]
            # MONAI’s DiceLoss produces [B, C] if batch=False, reduction='none'
            # but its implementation expects one-hot target
            # print(f'Target shape {target.shape}, logits shape {logits.shape}')
            dice_val_per_class = self.dice(logits, target)
            # average over classes to get [B]
            dice_per_sample = dice_val_per_class.view(B, -1).mean(dim=1)
            # print("Dice:", dice_per_sample.min(), dice_per_sample.max(), dice_per_sample.mean(), dice_per_sample.shape)

            # CE component: CrossEntropyLoss(reduction='none') yields [B, *spatial]
            ce_map = self.ce(logits, target.squeeze(1).long())
            ce_per_sample = ce_map.view(B, -1).mean(dim=1)

            # print("CE:", ce_map.min(), ce_map.max(), ce_map.mean(), ce_per_sample.shape)
            return dice_per_sample + ce_per_sample


def asym_exp_weighting(
    pred: Tensor,
    gt: Tensor,  
    tau: float, 
    s: float = 4
) -> Tensor:
    """
    Piecewise exponential with threshold tau and scale s:
      f(x) = exp( s*(x - tau) )  for x > tau
           = exp(-s*(tau - x))  for x <= tau
    Note: at x == tau, both give exp(0)==1.
    """
    tau_t = pred.new_tensor(tau)
    s_t   = pred.new_tensor(s)
    return where(
        gt > tau_t,
        exp( s_t * (tau_t - pred) ),
        exp( s_t * (pred - tau_t) )
    )


def asym_step_weighting(
    pred: Tensor,
    gt: Tensor,  
    tau: float, 
    s: float = 4
) -> Tensor:
    """
    Piecewise step function with threshold tau and scale s:
      f(x) = s    if (gt > tau and pred < gt) or (gt <= tau and pred > gt)  # High penalty cases
           = 1    otherwise  # Low penalty cases
    
    This creates the same asymmetric behavior as asym_exp_weighting but with constant values:
    - When gt > tau: penalizes under-prediction (pred < gt) with weight s, others get 1
    - When gt <= tau: penalizes over-prediction (pred > gt) with weight s, others get 1
    """
    tau_t = pred.new_tensor(tau)
    s_t = pred.new_tensor(s)
    one = pred.new_tensor(1.0)
    
    # Condition for high penalty (equivalent to where exponential would be high)
    high_penalty_condition = where(
        gt > tau_t,
        pred < tau_t,  # When gt > tau, penalize pred < gt (under-prediction)
        pred > tau_t   # When gt <= tau, penalize pred > gt (over-prediction)
    )
    
    return where(high_penalty_condition, s_t, one)


def thresholded_sigmoid_bce(
    pred: Tensor,
    target: Tensor,
    tau: float,
    s: float = 10.0,
) -> Tensor:
    """
    Shape-aligned BCE-style weighting that matches asym_step_weighting.

    Steps:
        1) Binarize GT: gt_bin = (target > tau)
        2) Smooth prediction: p = sigmoid(s*(pred - tau))
        3) BCE between p and gt_bin, then reduce over all non-batch dims to
                return a per-sample vector of shape [B], same as asym_step_weighting.

    Args:
        pred:   [B, ...] real-valued predictions (logits or scores)
        target: [B, ...] real-valued targets in [0,1]
        tau:    threshold for binarization/shift
        s:      sigmoid sharpness
        reduction: kept for API compatibility, but ignored (always returns [B])

    Returns:
        Tensor: per-sample values with shape [B]
"""
    # 1) binary GT (no grad)
    gt_bin = (target > tau).to(pred.dtype)
    # 2) shifted sigmoid
    p = s * (pred - tau)
    # 3) elementwise BCE (no reduction), then average over non-batch dims
    elem_bce = binary_cross_entropy_with_logits(p, gt_bin, reduction="none")

    return elem_bce


def thresholded_sigmoid_focal_loss(
    pred: Tensor,
    target: Tensor,
    tau: float,
    s: float = 10.0,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> Tensor:
    """
    Thresholded sigmoid focal loss using torchvision's implementation.

    Steps:
      1) Binarize target with threshold tau: gt_bin = (target > tau)
      2) Shift-and-scale logits: logits = s * (pred - tau)
      3) Apply sigmoid_focal_loss(logits, gt_bin, alpha, gamma, reduction)

    Args:
        pred:   Tensor of arbitrary shape [B, ...] (raw scores/logits-like values)
        target: Tensor of same shape with continuous targets in [0, 1]
        tau:    Threshold for binarization and logit shift
        s:      Scale controlling the sharpness around the threshold
        alpha:  Focal loss alpha balancing factor
        gamma:  Focal loss gamma focusing parameter
        reduction: 'none' | 'mean' | 'sum' (as in torchvision.ops.sigmoid_focal_loss)

    Returns:
        Tensor of the same shape as pred if reduction='none', else a scalar.
    """
    gt_bin = (target > tau).to(pred.dtype)
    logits = s * (pred - tau)
    return sigmoid_focal_loss(logits, gt_bin, alpha=alpha, gamma=gamma, reduction=reduction)


def unified_score_loss(
    predicted_segmentation: Tensor, 
    target_segmentation: Tensor,
    prediction: Tensor,
    metric_type: str = 'dice',  # 'dice' or 'surface'
    criterion: Callable = None,  # unused for probabilistic regressions
    num_classes: int = 4,
    return_scores: bool = False,
    # Extra diagnostics / API
    return_terms: bool = False,
    verbose: bool = False,
    use_heads: str = 'all',          # 'all' or 'aggregate'
    kappa_reg_weight: float = 0.01,  # 0 to disable kappa regularization
    class_head_weight: float = 0.1,  # weighting factor for auxiliary class-wise heads (aggregate head weight = 1.0)
    beta: float = 0.0,               # beta exponent for stop-grad weighting (0 => plain NLL, 1 => mu gradient independent of kappa)
    # Likelihood selection for regular vs beta regression
    likelihood: str = 'beta',        # 'beta' | 'gaussian' | 'laplace' | 'mse'
) -> Tensor:
    """
    Unified multi-head Beta regression loss with optional weighting for auxiliary class-wise heads.

    Previous version: single foreground aggregate head.
    Extension: allow multiple Beta heads predicting:
        Head 0: aggregate foreground score (mean over non-NaN foreground classes)
        Heads 1..F: class-wise scores for each foreground class (excluding background)

    Expected prediction tensor shapes:
        [B, P, 2]  where P = 1 + (num_classes-1) = num_classes (aggregate + each class)
        or legacy [B, P, 3] where the 3rd value (pi0) is ignored.

    Each head parameterized by (mu, kappa) with mu in (0,1), kappa>0.
    Per-head masking: if class dice is NaN (empty), that head is masked out.

    Args:
        predicted_segmentation: [B, 1, H, W] integer labels or logits-derived predictions.
        target_segmentation:    [B, 1, H, W] integer labels.
        prediction:             [B, P, 2] (mu,kappa) or [B, P, 3] (mu,kappa,pi0; pi0 ignored).
        metric_type:            'dice' or 'surface'.
        num_classes:            number of classes including background.
        return_scores:          if True, also return (mu, y_fg).
    (gap regularization removed)

    Additional weighting:
        The first head (index 0) is the aggregate foreground score and always receives weight 1.0.
        Each per-class auxiliary head (indexes 1..F) is multiplied by class_head_weight (default 0.1)
        before aggregation (a weighted mean over valid heads). Setting class_head_weight=1 restores
        the previous unweighted behavior. If use_heads == 'aggregate', only the aggregate head is used.

    Returns:
        If return_scores: (loss: [B,1], mu: [B,P], y_fg: [B]) else: loss: [B,1]
    """

    # Compute per-class metric [B, C]
    if metric_type == 'dice':
        per_class = DiceMetric(
            include_background=True, 
            reduction="none",
            num_classes=num_classes,
            ignore_empty=False
        )(predicted_segmentation, target_segmentation).detach()
    elif metric_type == 'surface':
        y_pred_oh = one_hot(predicted_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
        y_true_oh = one_hot(target_segmentation.squeeze(1), num_classes=num_classes).moveaxis(-1, 1)
        per_class = SurfaceDiceMetric(
            include_background=True, 
            reduction="none",
            class_thresholds=[3] * num_classes,
        )(y_pred_oh, y_true_oh).detach()
    else:
        raise ValueError(f"metric_type must be 'dice' or 'surface', got {metric_type}")

    # Foreground classes (exclude background idx 0)
    fg = per_class[..., 1:]                # [B, F]
    # Aggregate foreground score (mean over non-NaN foreground classes)
    y_fg = fg.nanmean(dim=-1)              # [B]
    # Build full target matrix: first aggregate, then per-class
    y_targets = cat([y_fg.unsqueeze(1), fg], dim=1)  # [B, 1+F] == [B, num_classes]

    # Validity mask per head
    valid_mask_agg = ~isnan(y_fg)                      # [B]
    valid_mask_classes = ~isnan(fg)                    # [B, F]
    valid_mask = cat([valid_mask_agg.unsqueeze(1), valid_mask_classes], dim=1)  # [B, 1+F]

    # Prepare prediction -> mu, scale (second channel). For Beta we name it kappa.
    # Note: Prediction head already applies sigmoid/softplus; here we only clamp.
    if prediction.dim() == 2:
        # Legacy single head flattened => make [B,1,2]
        prediction = prediction.unsqueeze(1)
    if prediction.size(-1) == 3:
        mu = prediction[..., 0]
        scale_param = prediction[..., 1]
    elif prediction.size(-1) == 2:
        mu = prediction[..., 0]
        scale_param = prediction[..., 1]
    else:
        raise AssertionError("prediction last dim must be 2 or 3 per head: [mu, kappa] or [mu, kappa, *_]")

    # Sanity: number of heads should match targets (aggregate + per-class)
    assert mu.shape[1] == y_targets.shape[1], \
        f"Prediction heads ({mu.shape[1]}) != targets ({y_targets.shape[1]})"

    # Safety clamps (model should already apply sigmoid/softplus)
    eps = 1e-6
    mu = mu.clamp(eps, 1 - eps)
    scale_param = scale_param.clamp_min(eps)

    # Target broadcast to [B, P]; we MUST eliminate NaNs before passing to Beta log_prob.
    # torch.clamp does NOT modify NaNs, and NaN * 0 == NaN, so prior code allowed NaNs to leak.
    # Strategy:
    #   1. Build validity mask (already computed below)
    #   2. Replace invalid (NaN) targets with neutral value 0.5 (center of Beta support)
    #   3. Clamp to (eps, 1-eps)
    #   4. Compute log_prob on safe targets
    #   5. Zero-out invalid positions (instead of multiplying NaN by 0 later)
    valid_mask_bp = valid_mask.to(mu.dtype)            # [B, P]
    neutral = mu.new_tensor(0.5)
    # Preserve original targets for optional return, only sanitize copy used for NLL
    y_bp = where(valid_mask_bp.bool(), y_targets.nan_to_num(neutral), neutral)
    y_bp = y_bp.clamp(eps, 1 - eps)                    # [B, P] safe

    # Choose likelihood branch (no silent fallbacks)
    like = str(likelihood).lower()
    if verbose:
        print(f"[unified_score_loss] metric={metric_type} likelihood={like} use_heads={use_heads} beta={beta} kappa_reg_weight={kappa_reg_weight} class_head_weight={class_head_weight}")

    base_per_head = None
    reg_per_head = None

    if like == 'beta':
        kappa = scale_param
        # Beta parameters (mean-precision: a=mu*kappa, b=(1-mu)*kappa)
        alpha_param = mu * kappa
        beta_param = (1.0 - mu) * kappa
        dist = Beta(alpha_param, beta_param)
        base = -dist.log_prob(y_bp)  # NLL

        # beta-NLL stop-grad weighting: multiply per-head/sample nll by kappa^{-beta} (detached)
        if beta != 0.0:
            weight = kappa.detach().pow(-beta)
            base = base * weight

        # Optional kappa regularization on valid heads
        if kappa_reg_weight and kappa_reg_weight > 0:
            k_typical = tensor(80, device=kappa.device, dtype=kappa.dtype)
            s = 1.0
            mu_ln = log(k_typical) + s**2   # ensures mode = k_typical
            reg = ((log(kappa) - mu_ln)**2 / (2*s**2)) + log(kappa)
            reg = reg * valid_mask_bp  # zero invalid
            reg_per_head = reg
            base = base + kappa_reg_weight * reg
        base_per_head = base

    elif like == 'betahomoscedastic':

        device, dtype = mu.device, mu.dtype
        kappa = torch.tensor(100, device=device, dtype=dtype)
        # kappa = torch.ones_like(scale_param.detach()) * 80.0
        # Beta parameters (mean-precision: a=mu*kappa, b=(1-mu)*kappa)
        alpha_param = mu * kappa
        beta_param = (1.0 - mu) * kappa
        dist = Beta(alpha_param, beta_param)
        base = -dist.log_prob(y_bp)  # NLL

        # # beta-NLL stop-grad weighting: multiply per-head/sample nll by kappa^{-beta} (detached)
        # if beta != 0.0:
        #     weight = kappa.detach().pow(-beta)
        #     base = base * weight

        # # Optional kappa regularization on valid heads
        # if kappa_reg_weight and kappa_reg_weight > 0:
        #     k_typical = tensor(80, device=kappa.device, dtype=kappa.dtype)
        #     s = 1.0
        #     mu_ln = log(k_typical) + s**2   # ensures mode = k_typical
        #     reg = ((log(kappa) - mu_ln)**2 / (2*s**2)) + log(kappa)
        #     reg = reg * valid_mask_bp  # zero invalid
        #     reg_per_head = reg
        #     base = base + kappa_reg_weight * reg
        base_per_head = base

    elif like == 'gaussian':
        sigma = scale_param
        var = sigma * sigma + eps
        base = 0.5 * (log(2 * torch.pi * var) + (y_bp - mu).pow(2) / var)
        if beta != 0.0:
            base = base * sigma.detach().pow(-beta)
        base_per_head = base
        # no default scale reg for gaussian (can be added later via kappa_reg_weight2)

    elif like == 'laplace':
        b = scale_param
        base = torch.log(2*b) + (y_bp - mu).abs() / b
        if beta != 0.0:
            base = base * b.detach().pow(-beta)
        base_per_head = base

    elif like == 'mse':
        base = (y_bp - mu).pow(2)
        base_per_head = base
        # ignore second channel in MSE; no weighting by beta

    else:
        raise ValueError(f"Unsupported likelihood '{likelihood}'. Expected one of ['beta', 'betahomoscedastic','gaussian','laplace','mse'].")

    # Zero-out invalid entries explicitly (avoid NaN * 0 propagation)
    base_per_head = where(valid_mask_bp.bool(), base_per_head, base_per_head.new_zeros(()))
    if reg_per_head is not None:
        reg_per_head = where(valid_mask_bp.bool(), reg_per_head, reg_per_head.new_zeros(()))
        per_head_loss = base_per_head  # already includes reg when beta branch added it
    else:
        per_head_loss = base_per_head
    loss = per_head_loss

    assert loss.isnan().sum() == 0, f"Loss contains NaN values: {loss.isnan().sum().item()}"

    # Weighting: aggregate head weight 1.0, each auxiliary class head weight = class_head_weight
    if loss.shape[1] > 1:
        head_weights = loss.new_full((loss.shape[1],), class_head_weight)
        head_weights[0] = 1.0
    else:
        head_weights = loss.new_tensor([1.0])

    # Broadcast weights to batch
    head_weights_b = head_weights.unsqueeze(0).expand_as(loss)
    weighted_loss = loss * head_weights_b

    if use_heads == 'aggregate':
        # Only aggregate head (ignore weighting of others entirely)
        agg_loss = weighted_loss[:, 0:1]
        agg_mask = valid_mask_bp[:, 0:1]
        denom = agg_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        loss = agg_loss.sum(dim=1, keepdim=True) / denom
    else:
        # Weighted mean over valid heads
        effective_weights = head_weights_b * valid_mask_bp
        denom = effective_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        loss = weighted_loss.sum(dim=1, keepdim=True) / denom

    # Prepare terms for optional diagnostics/logging (aggregated to [B,1])
    if return_terms:
        # Aggregate base and reg with same head weights/mask
        def aggregate(x: Tensor) -> Tensor:
            ew = head_weights_b * valid_mask_bp
            return (x * head_weights_b).sum(dim=1, keepdim=True) / ew.sum(dim=1, keepdim=True).clamp_min(1.0)

        terms = {}
        key = {'beta': 'nll', 'betahomoscedastic': 'nll', 'gaussian': 'nll', 'laplace': 'nll', 'mse': 'mse'}[like]
        terms[key] = aggregate(base_per_head).detach()
        if reg_per_head is not None:
            terms['kappa_reg'] = aggregate(reg_per_head).detach()
        # Also expose aggregate vs aux heads split
        terms['agg_head_loss'] = (loss.new_tensor(1.0) * (weighted_loss[:, 0:1] / valid_mask_bp[:, 0:1].clamp_min(1.0))).detach()
        if weighted_loss.shape[1] > 1:
            aux = weighted_loss[:, 1:]
            aux_mask = valid_mask_bp[:, 1:]
            aux_w = head_weights_b[:, 1:]
            terms['aux_heads_loss'] = (aux.sum(dim=1, keepdim=True) / (aux_w * aux_mask).sum(dim=1, keepdim=True).clamp_min(1.0)).detach()
        terms['primary'] = loss.detach()
        terms['valid_heads'] = valid_mask_bp.sum(dim=1, keepdim=True).detach()

    if return_scores and return_terms:
        pred_out = mu.detach() * valid_mask_bp
        return loss, terms, pred_out.nan_to_num(0), y_targets.nan_to_num(0)
    elif return_scores:
        pred_out = mu.detach() * valid_mask_bp
        return loss, pred_out.nan_to_num(0), y_targets.nan_to_num(0)
    elif return_terms:
        return loss, terms
    else:
        return loss