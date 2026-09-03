import os
import wandb
import shutil
import argparse
from time import time

import torch
import numpy as np
from omegaconf import OmegaConf

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from diffusers import AutoencoderKL
from accelerate import Accelerator
from accelerate.utils import set_seed

from datasets_prep import get_ir_dataset
from models import create_network, DecoderFeatureExtractor, TAESDWrapper, TAESDDecoderFeatureExtractor
from utils import EMA, cfm_loss, cfm_lpips_loss, cfm_lpl_loss
from torchdiffeq import odeint

from lpips import LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from elatentlpips import ELatentLPIPS

from ConFIG.conflictfree.utils import apply_gradient_vector, get_gradient_vector
from ConFIG.conflictfree.utils import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb

def sample_from_model(model, x_0, z_cond=None, steps=5, method="euler"):
    """
    x_0 : [B, C, H, W] (degraded latent for unconditional, noise for conditional)
    z_cond : conditioning latent (for conditional flow formulation only)
    """
    t = torch.linspace(0.0, 1.0, steps+1, device=x_0.device, dtype=x_0.dtype)
    def ode_func(t_scalar, x):
        B = x.shape[0]
        noise_labels = torch.full((B,), t_scalar, device=x.device, dtype=x.dtype)

        if z_cond is not None:
            x_in = torch.cat([x, z_cond], dim=1)
        else:
            x_in = x

        return model(noise_labels, x_in)

    fake_latents = odeint(ode_func, x_0, t, method=method, atol=1e-5, rtol=1e-5)
    return fake_latents  # [steps + 1, B, C, H, W]

def increase_scale_schedule(t, schedule_type):
        """
        t: shape (B,1,1,1)
        return: scale (B,1,1,1)
        """
        t = t + 1e-8

        if schedule_type == "linear":
            scale = t
        elif schedule_type == "linear_warmup":
            scale = torch.where(t < 0.5, torch.zeros_like(t), t - 0.5)
        else:
            raise ValueError("Select alpha increase schedule. [constant, lienar, linear_warmup]")

        return scale

def get_perceptual_weight(t, args):
    """
    timestep-based perceptual weight scheduling
    t: shape (B,1,1,1)
    returns: shape-compatible weight tensor or scalar
    """
    if args.lambda_schedule == "constant":
        return torch.ones_like(t) * args.lambda_percep

    if args.lambda_schedule.startswith("in_"):
        schedule_type = args.lambda_schedule[3:]
        scale = increase_scale_schedule(t, schedule_type)
        return scale * args.lambda_percep

    raise ValueError(f"Unknown lambda_schedule: {args.lambda_schedule}")
    
def gradient_projection(
    grad_1: torch.Tensor,
    grad_2: torch.Tensor,
    args):

    eps = 1e-8

    with torch.no_grad():
        g1g2 = torch.dot(grad_1, grad_2)
        if g1g2 <= 0:
            if args.projection_type == 'one_projection':
                or_grad_1 = grad_1 - grad_2 * g1g2 / (grad_2.norm() ** 2 + eps)
                return or_grad_1, grad_2
            else:
                or_grad_1 = grad_1 - grad_2 * g1g2 / (grad_2.norm() ** 2 + eps)
                or_grad_2 = grad_2 - grad_1 * g1g2 / (grad_1.norm() ** 2 + eps)
                return or_grad_1, or_grad_2
        else:
            return grad_1, grad_2  


def train(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    dtype = torch.float32
    set_seed(args.seed + accelerator.process_index)

    batch_size = args.batch_size

    train_dataset, valid_dataset = get_ir_dataset(args)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    model = create_network(args).to(device, dtype=dtype)
    if args.use_grad_checkpointing:
        model.set_gradient_checkpointing()

    if args.first_stage_model_type == "kl":
        first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    elif args.first_stage_model_type == "taesd":
        first_stage_model = TAESDWrapper(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    
    first_stage_model.eval()
    for param in first_stage_model.parameters():
        param.requires_grad = False

    lq_encoder = None
    if args.finetune_lq_encoder:
        if args.first_stage_model_type == "kl":
            lq_encoder = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
        elif args.first_stage_model_type == "taesd":
            lq_encoder = TAESDWrapper(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
 
        lq_encoder.train()
        for param in lq_encoder.parameters():
            param.requires_grad = True
                    
    if args.use_lpips_loss:
        elatentlpips_model = ELatentLPIPS(encoder="taesd", \
                                        augment='bg', \
                                        taesd_lpips_pth=args.taesd_lpips_pth,\
                                        taesd_vgg_pth=args.taesd_vgg_pth, \
                                        verbose=False).to(device, dtype=dtype).eval()
        for param in elatentlpips_model.parameters():
            param.requires_grad = False
    
    if args.use_lpl_loss:
        if args.first_stage_model_type == "kl":
            decoder_extractor = DecoderFeatureExtractor(first_stage_model.decoder).to(device, dtype=dtype)
            percep_keys = args.percep_keys
            percep_weights = args.percep_weights
        else:
            decoder_extractor = TAESDDecoderFeatureExtractor(first_stage_model.model.decoder).to(device, dtype=dtype)
            percep_keys = args.percep_keys
            percep_weights = args.percep_weights

            decoder_extractor.eval()
            for p in decoder_extractor.parameters():
                p.requires_grad = False

    # learning parameters
    param_groups = [{"params": list(model.parameters()), "lr": args.lr}]
    if args.finetune_lq_encoder and lq_encoder is not None:
        param_groups.append({"params": list(lq_encoder.parameters()), "lr": args.lr})
    
    # optimizer/scheduler
    optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, betas=(args.beta1, args.beta2))
    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)

    if args.lr_scheduler_type == 'CosineAnneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)
    elif args.lr_scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    train_dataloader, valid_dataloader, model, optimizer, scheduler = accelerator.prepare(train_dataloader, valid_dataloader, model, optimizer, scheduler)
    if args.finetune_lq_encoder and lq_encoder is not None:
        lq_encoder = accelerator.prepare(lq_encoder)

    def swap_ema():
        if hasattr(optimizer, "swap_parameters_with_ema"):
            optimizer.swap_parameters_with_ema(store_params_in_ema=True)
        else:
            optimizer.optimizer.swap_parameters_with_ema(store_params_in_ema=True)

    exp = args.exp
    if args.use_lpips_loss:
        parent_dir = "./saved_info/pcflow/{}/lfm-fr-elpips-loss".format(args.dataset)
    elif args.use_lpl_loss:
         parent_dir = "./saved_info/pcflow/{}/lfm-fr-lpl-loss".format(args.dataset)
    elif args.add_consistency_loss:
        parent_dir = "./saved_info/pcflow/{}/lfm-fr-consistency-loss".format(args.dataset)
    else:
        raise ValueError(f"Unknown loss function type")
    exp_path = os.path.join(parent_dir, exp)
    if accelerator.is_main_process:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            config_dict = vars(args)
            OmegaConf.save(config_dict, os.path.join(exp_path, "config.yaml"))
        if args.wandb_logging:
            if args.use_lpl_loss:
                wandb.init(entity="entity-name", project="lfm-fr-lpl-loss", name=args.exp, config=vars(args))
            elif args.use_lpips_loss:
                wandb.init(entity="entity-name", project="lfm-fr-elpips-loss", name=args.exp, config=vars(args))
            elif args.add_consistency_loss:
                wandb.init(entity="entity-name", project="lfm-fr-consistency-loss", name=args.exp, config=vars(args))
            else:
                raise ValueError(f"Unknown loss function type")

    if args.resume or os.path.exists(os.path.join(exp_path, "content.pth")):
        checkpoint_file = os.path.join(exp_path, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        model.load_state_dict(checkpoint["model_dict"])
        if args.finetune_lq_encoder and lq_encoder is not None and "lq_encoder_dict" in checkpoint:
            lq_encoder.load_state_dict(checkpoint["lq_encoder_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = checkpoint["global_step"]
        
        accelerator.print("=> resume checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint

    elif args.model_ckpt and os.path.exists(os.path.join(exp_path, args.model_ckpt)):
        checkpoint_file = os.path.join(exp_path, args.model_ckpt)
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)

        checkpoint_enc = torch.load(args.lq_encoder_ckpt)
        epoch = int(args.model_ckpt.split("_")[-1][:-4])

        init_epoch = epoch
        
        model.load_state_dict(checkpoint)
        lq_encoder.load_state_dict(checkpoint_enc)

        global_step = 0

        accelerator.print("=> loaded checkpoint (epoch {})".format(epoch))
        del checkpoint, checkpoint_enc
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    is_latent_data = True if "latent" in args.dataset else False
    log_steps = 0
        
    # remove warning messages 
    import warnings
    warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

    # initialize metrics    
    fid_metric = FrechetInceptionDistance(feature=2048).to(accelerator.device)
    lpips_metric = LPIPS(net='vgg').to(accelerator.device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(accelerator.device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(accelerator.device)  # images in [-1,1]

    start_time = time()

    for epoch in range(init_epoch, init_epoch + args.num_epoch + 1):

        # ----- train
        model.train()
        if args.finetune_lq_encoder and lq_encoder is not None:
            lq_encoder.train()
        if args.use_lpl_loss:
            decoder_extractor.eval()

        train_loss, train_loss_consistency, train_loss_percep = 0.0, 0.0, 0.0
        if args.use_lpl_loss:
            train_loss_percep_layers = {k: 0.0 for k in percep_keys}
        for iteration, (x_deg, x_gt) in enumerate(train_dataloader):
            x_deg = x_deg.to(device, dtype=dtype, non_blocking=True)
            x_gt = x_gt.to(device, dtype=dtype, non_blocking=True)
            
            optimizer.zero_grad()
            
            if is_latent_data:
                z_cond = x_deg * args.scale_factor
                z_1 = x_gt * args.scale_factor
            else:
                with torch.no_grad():
                    z_1 = first_stage_model.encode(x_gt).latent_dist.sample().mul_(args.scale_factor)
                if args.finetune_lq_encoder and lq_encoder is not None:
                    z_cond = lq_encoder.encode(x_deg).latent_dist.sample().mul_(args.scale_factor)
                else:
                    with torch.no_grad():
                        z_cond = lq_encoder.encode(x_deg).latent_dist.sample().mul_(args.scale_factor)

            if args.conditional:
                z_0 = torch.randn_like(z_1)
            else: 
                z_0 = z_cond + args.sigma * torch.randn_like(z_cond)

            with accelerator.autocast():
                loss_consistency = torch.zeros((), device=device)
                loss_percep = torch.zeros((), device=device)

                # t should be in [0, 1-dt] to allow for r = t + dt
                t = (1 - args.consistency_dt) * torch.rand((z_0.shape[0],1,1,1), device=device, dtype=dtype)

                if args.use_lpips_loss:
                    loss_consistency, loss_trajectory, alpha, loss_velocity, loss_percep = cfm_lpips_loss(model, z_0, z_1, t, z_cond, elatentlpips_model, args) 
                elif args.use_lpl_loss:
                    loss_consistency, loss_trajectory, alpha, loss_velocity, loss_percep, details = cfm_lpl_loss(model, z_0, z_1, t, z_cond, args, decoder_extractor, percep_keys, percep_weights)
                else:
                    loss_consistency, loss_trajectory, alpha, loss_velocity = cfm_loss(model, z_0, z_1, t, z_cond, args)
                
                # perceptual loss weight
                percep_weight = get_perceptual_weight(t, args)
                loss_percep = (percep_weight * loss_percep).mean()
                loss = loss_consistency + loss_percep

            # conflict-free gradient alignment
            if args.use_conflict_free:
                model_grads = []
                lq_grads = []

                optimizer.zero_grad()
                accelerator.backward(loss_consistency, retain_graph=True)
                model_grads.append(get_gradient_vector(model))
                lq_grads.append(get_gradient_vector(lq_encoder))

                optimizer.zero_grad()
                accelerator.backward(loss_percep)
                model_grads.append(get_gradient_vector(model))
                lq_grads.append(get_gradient_vector(lq_encoder))

                m_grad_cfm, m_grad_p = model_grads
                lq_grad_cfm, lq_grad_p = lq_grads

                if args.projection_type == 'one_projection':
                
                    if args.one_projected_gradient == 'cfm_gradient':
                        m_lq_grad_cfm = torch.cat([m_grad_cfm, lq_grad_cfm])
                        grad_p_m_lq = torch.cat([m_grad_p, lq_grad_p])

                        m_or_lq_grad_cfm, m_lq_grad_p = gradient_projection(m_lq_grad_cfm, grad_p_m_lq, args)
                        
                        m_or_grad_cfm, lq_or_grad_cfm = torch.split(m_or_lq_grad_cfm, [m_grad_cfm.size(0), lq_grad_cfm.size(0)])
                        m_grad_p, lq_grad_p = torch.split(m_lq_grad_p, [m_grad_p.size(0), lq_grad_p.size(0)])

                        m_final_grad = m_or_grad_cfm + m_grad_p
                        lq_final_grad = lq_or_grad_cfm + lq_grad_p

                    elif args.one_projected_gradient == 'perceptual_gradient':
                        m_lq_grad_cfm = torch.cat([m_grad_cfm, lq_grad_cfm])
                        grad_p_m_lq = torch.cat([m_grad_p, lq_grad_p])

                        m_lq_or_grad_p, m_lq_grad_cfm = gradient_projection(grad_p_m_lq, m_lq_grad_cfm, args)

                        m_grad_cfm, lq_grad_cfm = torch.split(m_lq_grad_cfm, [m_grad_cfm.size(0), lq_grad_cfm.size(0)])
                        m_or_grad_p, lq_or_grad_p = torch.split(m_lq_or_grad_p, [m_grad_p.size(0), lq_grad_p.size(0)])

                        m_final_grad = m_grad_cfm + m_or_grad_p
                        lq_final_grad = lq_grad_cfm + lq_or_grad_p

                apply_gradient_vector(model, m_final_grad)
                apply_gradient_vector(lq_encoder, lq_final_grad)
            else:
                accelerator.backward(loss)

            optimizer.step()

            train_loss += loss.item()
            train_loss_consistency += loss_consistency.item() 
            train_loss_percep += loss_percep.item()

            if args.use_lpl_loss:
                for k in percep_keys:
                    train_loss_percep_layers[k] += details[k]["weighted_loss"]
            
            global_step += 1
            log_steps += 1

            if iteration % 100 == 0:
                if accelerator.is_main_process:
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time) 
                    accelerator.print(
                        "epoch {} iteration{}, Loss: {}, Consistency Loss: {}, Perceptual Loss:{}, Train Steps/Sec: {:.2f}".format(
                            epoch, iteration, loss.item(), loss_consistency.item(), loss_percep.item(), steps_per_sec
                        )
                    )
                    if args.use_lpl_loss:
                        loss_percep_info = ", ".join([f"{k}: {v['weighted_loss']:.4f}" for k, v in details.items()])
                        accelerator.print(f"    Perceptual Loss Details - {loss_percep_info}")
                       
                    # reset monitoring variables
                    log_steps = 0
                    start_time = time()

        if not args.no_lr_decay:
            scheduler.step()

        avg_train_loss_consistency = train_loss_consistency/len(train_dataloader)
        avg_train_loss_percep = train_loss_percep/len(train_dataloader)
        avg_train_loss = train_loss / len(train_dataloader)

        avg_train_loss_percep_layers = {}
        if args.use_lpl_loss:
            for k in percep_keys:
                avg_train_loss_percep_layers[k] = train_loss_percep_layers[k]/len(train_dataloader)
                    
        if accelerator.is_main_process:
            if epoch % args.plot_every == 0:
                if args.use_ema:
                    swap_ema()
                
                model.eval()
                
                with torch.no_grad():
                    z0_sample = z_0[:4]
                    zcond_sample = z_cond[:4]

                    if args.conditional:
                        rand = torch.randn_like(z0_sample)
                        fake_sample = sample_from_model(model, rand, z_cond=zcond_sample)[-1]
                    else:
                        z_source = zcond_sample + args.sigma * torch.randn_like(zcond_sample)
                        fake_sample = sample_from_model(model, z_source)[-1]

                    fake_image = first_stage_model.decode((fake_sample / args.scale_factor).float()).sample
                
                if args.use_ema:
                    swap_ema()
                
                model.train()
                
                torchvision.utils.save_image(
                    fake_image,
                    os.path.join(exp_path, "image_epoch_{}.png".format(epoch)),
                )

            # ----- validation
            model.eval()
            if args.finetune_lq_encoder and lq_encoder is not None:
                lq_encoder.eval()
            if args.use_lpl_loss:
                decoder_extractor.eval()

            if args.use_ema:
                swap_ema()
            
            fid_metric.reset()
            psnr_metric.reset()
            ssim_metric.reset()
            lpips_values = []
            
            val_loss = 0.0
            val_loss_consistency = 0.0
            val_loss_percep = 0.0
            if args.use_lpl_loss:
                valid_loss_percep_layers = {k: 0.0 for k in percep_keys}

            with torch.no_grad():
                for val_x_deg, val_x_gt in valid_dataloader:
                    val_x_deg = val_x_deg.to(device, dtype=dtype)
                    val_x_gt = val_x_gt.to(device, dtype=dtype)
                    
                    z_1_pred = None

                    z_1 = first_stage_model.encode(val_x_gt).latent_dist.sample().mul_(args.scale_factor)
                    if args.finetune_lq_encoder and lq_encoder is not None:
                        z_cond = lq_encoder.encode(val_x_deg).latent_dist.sample().mul_(args.scale_factor)
                    else:
                        z_cond = lq_encoder.encode(val_x_deg).latent_dist.sample().mul_(args.scale_factor)

                    if args.conditional:
                        z_0 = torch.randn_like(z_1)
                    else: 
                        z_0 = z_cond + args.sigma * torch.randn_like(z_cond)

                    with accelerator.autocast():   
                        loss_consistency = torch.zeros((), device=device)
                        loss_percep = torch.zeros((), device=device)

                        # t should be in [0, 1-dt] to allow for r = t + dt
                        t = (1 - args.consistency_dt) * torch.rand((z_0.shape[0],1,1,1), device=device, dtype=dtype)

                        if args.use_lpips_loss:
                            loss_consistency, loss_trajectory, alpha, loss_velocity, loss_percep = cfm_lpips_loss(model, z_0, z_1, t, z_cond, elatentlpips_model, args) 
                        elif args.use_lpl_loss:
                            loss_consistency, loss_trajectory, alpha, loss_velocity, loss_percep, details = cfm_lpl_loss(model, z_0, z_1, t, z_cond, args, decoder_extractor, percep_keys, percep_weights)
                        else:
                            loss_consistency, loss_trajectory, alpha, loss_velocity = cfm_loss(model, z_0, z_1, t, z_cond, args)
                
                        # perceptual loss weight 
                        percep_weight = get_perceptual_weight(t, args)
                        loss_percep = (percep_weight * loss_percep).mean()
                        loss = loss_consistency + loss_percep

                        if z_1_pred is None:
                            if args.conditional:
                                z_1_pred = sample_from_model(model, z_0, z_cond=z_cond, steps=args.consistency_k_steps, method=args.method)[-1]
                            else:
                                z_1_pred = sample_from_model(model, z_0, steps=args.consistency_k_steps, method=args.method)[-1]

                    val_loss += loss.item()
                    val_loss_consistency += loss_consistency.item()
                    val_loss_percep += loss_percep.item()

                    if args.use_lpl_loss:
                        for k in percep_keys:
                            valid_loss_percep_layers[k] += details[k]["weighted_loss"]

                    val_x_pred = first_stage_model.decode((z_1_pred / args.scale_factor).float()).sample
                    val_x_pred = torch.clamp(val_x_pred, min=0.0, max=1.0) # clamping to [0,1]
                    
                    # update metrics 
                    lpips_val = lpips_metric(val_x_pred * 2 - 1, val_x_gt * 2 - 1)
                    lpips_values.append(lpips_val.mean().item())

                    fid_metric.update((val_x_pred * 255).clamp(0,255).to(torch.uint8), real=False)
                    fid_metric.update((val_x_gt * 255).clamp(0,255).to(torch.uint8), real=True)
                    psnr_metric.update(val_x_pred, val_x_gt)
                    ssim_metric.update(val_x_pred, val_x_gt)

            # compute metrics
            lpips_value = np.mean(lpips_values)
            fid_value = float(fid_metric.compute().item())
            psnr_value = float(psnr_metric.compute().item())
            ssim_value = float(ssim_metric.compute().item())

            if args.use_ema:
                swap_ema()

            avg_val_loss_consistency = val_loss_consistency/len(valid_dataloader)
            avg_val_loss_percep = val_loss_percep/len(valid_dataloader)
            avg_val_loss = val_loss/len(valid_dataloader)

            avg_valid_loss_percep_layers = {}
            if args.use_lpl_loss:
                for k in percep_keys:
                    avg_valid_loss_percep_layers[k] = valid_loss_percep_layers[k]/len(valid_dataloader)
                
            if args.wandb_logging:
                log_dict = {
                    "train_loss": avg_train_loss,
                    "train_loss_consistency": avg_train_loss_consistency,
                    "train_loss_percep": avg_train_loss_percep,
                    **{f"train_loss_percep_{k}": v for k, v in avg_train_loss_percep_layers.items()},
                    "val_loss": avg_val_loss,
                    "val_loss_consistency": avg_val_loss_consistency,
                    "val_loss_percep": avg_val_loss_percep,
                    **{f"val_loss_percep_{k}": v for k, v in avg_valid_loss_percep_layers.items()},
                    "val_fid": fid_value,
                    "val_lpips": lpips_value,
                    "val_psnr": psnr_value,
                    "val_ssim": ssim_value,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
                wandb.log(log_dict)

            epoch_summary = (f"[Epoch {epoch}] Train Loss={avg_train_loss:.4f} (consist={avg_train_loss_consistency:.4f}, percep={avg_train_loss_percep:.4f}) | "
                            f"Val Loss={avg_val_loss:.4f} | FID={fid_value:.2f} | LPIPS={lpips_value:.3f} | PSNR={psnr_value:.3f} | SSIM={ssim_value:.3f}")
            accelerator.print(epoch_summary)

            if epoch % args.save_ckpt_every == 0:
                if args.use_ema:
                    swap_ema()

                torch.save(
                    model.state_dict(),
                    os.path.join(exp_path, "model_{}.pth".format(epoch)),
                )
                if args.finetune_lq_encoder and lq_encoder is not None:
                    torch.save(
                        lq_encoder.state_dict(),
                        os.path.join(exp_path, "lq_encoder_{}.pth".format(epoch)),
                    )
                if args.use_ema:
                    swap_ema()

            if args.save_content:
                if epoch % args.save_content_every == 0:
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        "model_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    if args.finetune_lq_encoder and lq_encoder is not None:
                        content["lq_encoder_dict"] = lq_encoder.state_dict()
                    torch.save(content, os.path.join(exp_path, "content.pth"))

        accelerator.wait_for_everyone()

    if args.wandb_logging:
        if accelerator.is_main_process:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("pcflow training parameters")
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")
    parser.add_argument("--mode", type=str, default="train", help="train mode")

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None, help="Model ckpt to init from")

    parser.add_argument(
        "--model_type",
        type=str,
        default="adm",
        help="model_type",
        choices=[
            "adm",
            "ncsn++",
            "ddpm++",
            "DiT-B/2",
            "DiT-L/2",
            "DiT-L/4",
            "DiT-XL/2",
            "lunet",
        ],
    )
    parser.add_argument("--image_size", type=int, default=512, help="size of image")
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsample rate of input image by the autoencoder",
    )
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=3, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of model")
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=(16,),
        help="resolution of applying attention",
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        type=int,
        default=(1, 1, 2, 2, 4, 4),
        help="channel mult",
    )
    parser.add_argument("--n_mid_blocks", type=int, default=3, help="number of mid blocks")
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--label_dim", type=int, default=0, help="label dimension, 0 if unconditional")
    parser.add_argument(
        "--augment_dim",
        type=int,
        default=0,
        help="dimension of augmented label, 0 if not used",
    )
    parser.add_argument("--num_classes", type=int, default=None, help="num classes")
    parser.add_argument(
        "--label_dropout",
        type=float,
        default=0.0,
        help="Dropout probability of class labels for classifier-free guidance",
    )

    # original ADM
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--use_origin_adm", action="store_true")
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument("--num_heads", type=int, default=4, help="number of head")
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")

    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")

    # training
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--train_deg_datadir",type=str, default=None,  help="root dir for train dataset (deg image)")
    parser.add_argument("--train_gt_datadir", type=str, default=None,  help="root dir for train dataset (gt image)")
    parser.add_argument("--valid_deg_datadir", type=str, default=None, help="root dir for valid dataset (deg image)")
    parser.add_argument("--valid_gt_datadir", type=str, default=None, help="root dir for valid dataset (gt image)")
    parser.add_argument(
        "--use_grad_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing for mem saving",
    )
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "bf16", "fp16"], help="mixed precision mode")

    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=1200)

    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate g")

    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay for optimizer")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument("--use_ema", action="store_true", default=False, help="use EMA or not")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate for EMA")

    parser.add_argument("--save_ckpt_every", type=int, default=25, help="save ckpt every x epochs")
    parser.add_argument("--save_content", action="store_true", default=False)
    parser.add_argument("--save_content_every", type=int, default=10, help="save content for resuming every x epochs")
    parser.add_argument("--plot_every", type=int, default=5, help="plot every x epochs")

    # additional features (e.g., conditional/unconditional inputs, addition of noise, wandb)
    parser.add_argument('--conditional', action='store_true',
                    help='If set, the flow model is conditioned on either y or the posterior mean predictor. '
                            'Applies only to the stage "flow".')
    parser.add_argument("--sigma", type=float, default=0.0, help="gaussian noise std")
    parser.add_argument("--wandb_logging", action='store_true', help="wandb logging")
    
    # degradation type
    parser.add_argument('--degradation', type=str, required=True,
                        choices=['sr_bicubic_x8_gaussian_noise_005',
                                 'gaussian_noise_035',
                                 'colorization_gaussian_noise_025',
                                 'random_inpainting_gaussian_noise_01',
                                 'difface'],
                        help='The degradation type.')

    parser.add_argument("--first_stage_model_type", type=str, default="kl", choices=["kl", "taesd"], help="Type of first stage model")
    
    # lq_encoder
    parser.add_argument("--finetune_lq_encoder", action="store_true",
                        help="Use a separate trainable encoder for LQ images (finetuned from pretrained weights), "
                             "while HQ encoder stays frozen. Inspired by ELIR's teacher-student encoder setup.")
    parser.add_argument("--lq_encoder_ckpt", type=str, default=None,
                        help="Path for encoder model weight file")

    # consistency learning parameters
    parser.add_argument("--add_consistency_loss", action='store_true',
                        help="Add consistency flow matching loss to improve trajectory consistency")
    parser.add_argument("--consistency_alpha", type=float, default=0.001,
                        help="Weight for velocity consistency term within consistency loss (default: 0.001)")
    parser.add_argument("--consistency_k_steps", type=int, default=3,
                        help="Number of segments to divide the trajectory into (default: 5)")
    parser.add_argument("--consistency_dt", type=float, default=0.05,
                        help="Timestep delta for consistency computation (default: 0.05)")

    # perceptual loss parameters
    parser.add_argument("--use_lpl_loss", action='store_true', help="addition of perceptual loss")
    parser.add_argument("--lambda_percep", type=float, default=1.0, help="weight for perceptual loss")
    parser.add_argument("--percep_keys", nargs="+", type=str, default=["mid_block", "up_block_0", "up_block_1", "up_block_2", "up_block_3", "conv_out"], help="keys for perceptual loss")
    parser.add_argument("--percep_weights", nargs="+", type=float, default=[1, 1, 1, 1, 1, 1], help="weights for perceptual loss")

    parser.add_argument("--use_lpips_loss", action="store_true", help="Use E-Latent LPIPS as a distillation loss component")
    parser.add_argument("--lpips_normalization", type=bool, default=True, help="Activation for normalization")
    parser.add_argument("--taesd_lpips_pth",type=str, default=None,  help="root dir for latentLPIPS of taesd")
    parser.add_argument("--taesd_vgg_pth",type=str, default=None,  help="root dir for latentLPIPS of taesd")

    # sampling
    parser.add_argument("--method", type=str, default="euler", help="ODE solver")
    parser.add_argument("--steps", type=int, default=3, help="number of sampling steps")

    # conflict-free gradient alignment
    parser.add_argument("--use_conflict_free",action="store_true",  help="use modified gradient vector loss")
    parser.add_argument("--projection_type", type=str, default="off", choices=["one_projection", "two_projection"], help="Choose projection type")
    parser.add_argument("--one_projected_gradient", type=str, default="off", choices=["perceptual_gradient", "cfm_gradient", "off"], help="Choose projected gradient")

    # scheduler
    parser.add_argument("--lr_scheduler_type", type=str, default="CosineAnneal", choices=["CosineAnneal", "StepLR"], help="Choose projected gradient")
    parser.add_argument("--lambda_schedule", type=str, default="constant", choices=["constant", "in_linear", "in_linear_warmup"], help="Choose schedule type")
    
    args = parser.parse_args()
    train(args)