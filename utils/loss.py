import torch 
import torch.nn.functional as F


def cfm_loss(model, z_0, z_1, t, z_cond, args):
    """
    Latent Consistency Flow Matching (LCFM).

    Args:
        model: Flow matching model
        z_1: Clean latent (target)
        z_0: Starting latent (degraded + noise or pure noise)
        z_cond: Conditioning latent (for conditional models)
        args: Training arguments containing consistency hyperparameters

    Returns:
        loss_consistency (LCFM)
    """
    sigma_min = 1e-5
    alpha = args.consistency_alpha
    K = args.consistency_k_steps
    dt = args.consistency_dt

    bs = z_0.shape[0]
    device = z_0.device
    dtype = z_0.dtype

    # split trajectory into K segments
    segments = torch.linspace(0, 1, K + 1, device=device, dtype=dtype)
    seg_indices = torch.searchsorted(segments, t.view(-1), side="left").clamp(min=1)
    seg_ends = segments[seg_indices].view(bs, 1, 1, 1)

    # compute z_t at current timestep (same interpolation as main training)
    z_t = t * z_1 + (1 - (1 - sigma_min) * t) * z_0

    # compute velocity at t
    if args.conditional:
        z_in = torch.cat([z_t, z_cond], dim=1)
    else:
        z_in = z_t
    v0 = model(t.squeeze(), z_in)

    # compute z_r at next timestep r = t + dt
    r = t + dt
    z_r = r * z_1 + (1 - (1 - sigma_min) * r) * z_0

    # compute velocity at r (with no gradient for target)
    with torch.no_grad():
        if args.conditional:
            z_r_in = torch.cat([z_r, z_cond], dim=1)
        else:
            z_r_in = z_r
        v0_ = model(r.squeeze(), z_r_in)

    # compute predicted endpoints at segment boundary
    f0 = z_t + (seg_ends - t) * v0

    # from r: f0_ = z_r + (seg_ends - r) * v0_ (if r < seg_ends, else use actual endpoint)
    # compute actual endpoint for samples where r >= seg_ends
    z_seg_end = seg_ends * z_1 + (1 - (1 - sigma_min) * seg_ends) * z_0

    r_less = (r < seg_ends).float()
    f0_ = r_less * (z_r + (seg_ends - r) * v0_) + (1 - r_less) * z_seg_end

    # consistency loss
    loss_trajectory = F.mse_loss(f0, f0_.detach())
    loss_velocity = F.mse_loss(v0, v0_.detach())

    loss_consistency = loss_trajectory + alpha * loss_velocity

    return loss_consistency


def cfm_lpips_loss(model, z_0, z_1, t, z_cond, elatentlpips_model, args):
    """
    Latent Consistency Flow Matching (LCFM) with Latent Consistency Perceptual Loss (LCPL: External Perceptual Network).

    Args:
        model: Flow matching model
        z_1: Clean latent (target)
        z_0: Starting latent (degraded + noise or pure noise)
        z_cond: Conditioning latent (for conditional models)
        args: Training arguments containing consistency hyperparameters

    Returns:
        loss_consistency (LCFM), loss_percep (LCPL)
    """
    sigma_min = 1e-5
    alpha = args.consistency_alpha
    K = args.consistency_k_steps    
    dt = args.consistency_dt        

    bs = z_0.shape[0]
    device = z_0.device
    dtype = z_0.dtype

    # split trajectory into K segments
    segments = torch.linspace(0, 1, K + 1, device=device, dtype=dtype)
    seg_indices = torch.searchsorted(segments, t.view(-1), side="left").clamp(min=1)
    seg_ends = segments[seg_indices].view(bs, 1, 1, 1)

    # compute z_t at current timestep (same interpolation as main training)
    z_t = t * z_1 + (1 - (1 - sigma_min) * t) * z_0

    # compute velocity at t
    if args.conditional:
        z_in = torch.cat([z_t, z_cond], dim=1)
    else:
        z_in = z_t
    v0 = model(t.squeeze(), z_in)

    # compute z_r at next timestep r = t + dt
    r = t + dt
    z_r = r * z_1 + (1 - (1 - sigma_min) * r) * z_0

    # compute velocity at r (with no gradient for target)
    with torch.no_grad():
        if args.conditional:
            z_r_in = torch.cat([z_r, z_cond], dim=1)
        else:
            z_r_in = z_r
        v0_ = model(r.squeeze(), z_r_in)

    # compute predicted endpoints at segment boundary
    f0 = z_t + (seg_ends - t) * v0

    # from r: f0_ = z_r + (seg_ends - r) * v0_ (if r < seg_ends, else use actual endpoint)
    # compute actual endpoint for samples where r >= seg_ends
    z_seg_end = seg_ends * z_1 + (1 - (1 - sigma_min) * seg_ends) * z_0

    r_less = (r < seg_ends).float()
    f0_ = r_less * (z_r + (seg_ends - r) * v0_) + (1 - r_less) * z_seg_end

    # perceptual loss
    # loss_percep = elatentlpips_model(f0, f0_.detach(), normalize=args.lpips_normalization).mean()
    
    loss_percep = elatentlpips_model(f0, f0_.detach(), normalize=args.lpips_normalization).mean()

    # consistency loss
    loss_trajectory = F.mse_loss(f0, f0_.detach())
    loss_velocity = F.mse_loss(v0, v0_.detach())

    loss_consistency = loss_trajectory + alpha * loss_velocity

    return loss_consistency, loss_percep


def cfm_lpl_loss(model, z_0, z_1, t, z_cond, args, decoder_extractor, percep_keys, percep_weights):
    """
    Latent Consistency Flow Matching (LCFM) with Latent Consistency Perceptual Loss (LCPL: Internal Perceptual Network).

    Args:
        model: Flow matching model
        z_1: Clean latent (target)
        z_0: Starting latent (degraded + noise or pure noise)
        z_cond: Conditioning latent (for conditional models)
        args: Training arguments containing consistency hyperparameters

    Returns:
        loss_consistency (LCFM), loss_percep (LCPL)
    """
    sigma_min = 1e-5
    alpha = args.consistency_alpha
    K = args.consistency_k_steps   
    dt = args.consistency_dt

    bs = z_0.shape[0]
    device = z_0.device
    dtype = z_0.dtype

    # split trajectory into K segments
    segments = torch.linspace(0, 1, K + 1, device=device, dtype=dtype)
    seg_indices = torch.searchsorted(segments, t.view(-1), side="left").clamp(min=1)
    seg_ends = segments[seg_indices].view(bs, 1, 1, 1)

    # compute z_t at current timestep (same interpolation as main training)
    z_t = t * z_1 + (1 - (1 - sigma_min) * t) * z_0

    # compute velocity at t
    if args.conditional:
        z_in = torch.cat([z_t, z_cond], dim=1)
    else:
        z_in = z_t
    v0 = model(t.squeeze(), z_in)

    # compute z_r at next timestep r = t + dt
    r = t + dt
    z_r = r * z_1 + (1 - (1 - sigma_min) * r) * z_0

    # compute velocity at r (with no gradient for target)
    with torch.no_grad():
        if args.conditional:
            z_r_in = torch.cat([z_r, z_cond], dim=1)
        else:
            z_r_in = z_r
        v0_ = model(r.squeeze(), z_r_in)

    # compute predicted endpoints at segment boundary
    f0 = z_t + (seg_ends - t) * v0

    # from r: f0_ = z_r + (seg_ends - r) * v0_ (if r < seg_ends, else use actual endpoint)
    # compute actual endpoint for samples where r >= seg_ends
    z_seg_end = seg_ends * z_1 + (1 - (1 - sigma_min) * seg_ends) * z_0

    r_less = (r < seg_ends).float()
    f0_ = r_less * (z_r + (seg_ends - r) * v0_) + (1 - r_less) * z_seg_end

    # perceptual loss
    _, feat0 = decoder_extractor(f0 / args.scale_factor, return_features=True)
    with torch.no_grad():
        _, feat0_ = decoder_extractor(f0_ / args.scale_factor, return_features=True)
    loss_percep, details = lpl_loss(feat0, feat0_, keys=percep_keys, weights=percep_weights)

    # consistency loss
    loss_trajectory = F.mse_loss(f0, f0_.detach())
    loss_velocity = F.mse_loss(v0, v0_.detach())

    loss_consistency = loss_trajectory + alpha * loss_velocity

    return loss_consistency, loss_percep, details


def normalize_feature(feat, eps=1e-10):
    """ 
    Per-Channel Normalization.
    """
    mean = feat.mean(dim=[2, 3], keepdim=True)
    var = feat.var(dim=[2, 3], keepdim=True, unbiased=False)
    feat_norm = (feat - mean) / torch.sqrt(var + eps)
    return feat_norm


def lpl_loss(feats_pred, feats_gt, keys, weights=None):
    """
    feats_pred: list of feature maps from predicted output
    feats_gt:   list of feature maps from ground truth
    weights:    list of float weights for each feature level
    """
    feats_pred_lst = [feats_pred[k] for k in keys]
    feats_gt_lst = [feats_gt[k] for k in keys]

    if weights is None:
        weights = [1.0 / len(feats_pred_lst)] * len(feats_pred_lst)
    else:
        weights = [w / sum(weights) for w in weights]

    total_loss = 0.0
    loss_dict = {}
    
    for i, (k, fp, fg) in enumerate(zip(keys, feats_pred_lst, feats_gt_lst)):
        fp_norm = normalize_feature(fp)
        fg_norm = normalize_feature(fg)
        loss = F.mse_loss(fp_norm, fg_norm)
        weighted_loss = weights[i] * loss
        total_loss += weighted_loss

        loss_dict[k] = {
            "mse": loss.item(),
            "weight": weights[i],
            "weighted_loss": weighted_loss.item()
        }
        
    return total_loss, loss_dict