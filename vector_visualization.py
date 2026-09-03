import os
import shutil
import argparse

import torch
import numpy as np
from omegaconf import OmegaConf

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Subset

from accelerate import Accelerator
from accelerate.utils import set_seed

from datasets_prep import get_ir_dataset
from models import create_network, DecoderFeatureExtractor, TAESDWrapper, TAESDDecoderFeatureExtractor
from utils import EMA, cfm_loss, cfm_lpips_loss, cfm_lpl_loss, t_to_logsnr, ratio, cosine_sim 
from torchdiffeq import odeint

import itertools
from tqdm import tqdm
import bitsandbytes as bnb # debug for out of memory

from elatentlpips import ELatentLPIPS

from ConFIG.conflictfree.utils import apply_gradient_vector, get_gradient_vector
from ConFIG.conflictfree.utils import *

torch.backends.cudnn.benchmark = True


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

def decompose_vector(ref_grad, grad_v, grad_t):
    unit = ref_grad / (ref_grad.norm() + 1e-8)
    modi_v = grad_v - torch.dot(grad_v, unit) * unit
    modi_t = grad_t - torch.dot(grad_t, unit) * unit
    return modi_v, modi_t

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
        
    
def calculate_cosine_similarity(args):

    assert torch.cuda.is_available(), "Calculating cosine similarity currently requires at least one GPU."
    
    if args.visualization_save_path is None:
        raise ValueError("Saving directory must be re-specified")
    
    os.makedirs(args.visualization_save_path, exist_ok=True)
    
    sub_dirs = ["cosine_similarity", "ratio"]
    weight_folder_name = str(args.weight_folder).split('/')
    weight_folder_name = weight_folder_name[-1]

    paths = {}
    for sub in sub_dirs:
        path = os.path.join(args.visualization_save_path, weight_folder_name, sub)
        os.makedirs(path, exist_ok=True)
        paths[sub] = path

    OmegaConf.save(OmegaConf.create(vars(args)), os.path.join(args.visualization_save_path, weight_folder_name, "config.yaml"))

    cosine_save_path   = paths["cosine_similarity"]
    ratio_save_path    = paths["ratio"]

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device = accelerator.device
    dtype = torch.float32
    set_seed(args.seed + accelerator.process_index)

    batch_size = args.batch_size

    train_dataset, valid_dataset = get_ir_dataset(args)

    if args.train_len_control:
        train_dataset = Subset(train_dataset, range(args.train_length))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    model = create_network(args).to(device, dtype=dtype)

    lq_first_stage = TAESDWrapper(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    lq_encoder = lq_first_stage

    if args.first_stage_model_type == "kl":
        first_stage_model = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
    elif args.first_stage_model_type == "taesd":
        first_stage_model = TAESDWrapper(args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)

    first_stage_model.eval()
    if args.use_lpips_loss: 
        elatentlpips_model = ELatentLPIPS(encoder="taesd", \
                                        augment='bg', \
                                        taesd_lpips_pth=args.taesd_lpips_pth,\
                                        taesd_vgg_pth=args.taesd_vgg_pth, \
                                        verbose=False).to(device, dtype=dtype).eval()
        for param in elatentlpips_model.parameters():
            param.requires_grad = False
        accelerator.print("ELatentLPIPS model initialized.".center(60,'-'))
    
    if args.use_lpl_loss or args.conflict_free_perceptual_type == "lpl":
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

    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay) 

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)
    train_dataloader, valid_dataloader, model, _, _ = accelerator.prepare(train_dataloader, valid_dataloader, model, optimizer, scheduler)

    # list up all the weights
    ckpt_all_path_list = os.listdir(args.weight_folder)
    ckpt_path_list = [i for i in ckpt_all_path_list if i.endswith('.pth') and i.startswith('model')]
    ckpt_path_list = sorted(ckpt_path_list)
    ckpt_enc_path_list = [i for i in ckpt_all_path_list if i.endswith('.pth') and i.startswith('lq_encoder')]
    ckpt_enc_path_list = sorted(ckpt_enc_path_list)
    assert len(ckpt_enc_path_list) != 0 and len(ckpt_path_list) != 0, "Model checkpoint (model weight or encoder weight) is not in the given folder."
    assert len(ckpt_path_list) == len(ckpt_enc_path_list), "Number of encoder checkpoints does not match number of main model checkpoints"

    num_bins = args.bin_numb
    
    if args.time_step_type == 'snr':
        t_all = torch.linspace(0.0, 1.0, 2000, device=device)
        logsnr_all = t_to_logsnr(t_all)

        logsnr_bins = torch.linspace(
            logsnr_all.min(),
            logsnr_all.max(),
            num_bins + 1)
        
        logsnr_centers = 0.5 * (logsnr_bins[:-1] + logsnr_bins[1:])
    else:
        t_bins = torch.linspace(0, 1, num_bins)
    
    for i, (ckpt, enc_ckpt) in enumerate(zip(ckpt_path_list, ckpt_enc_path_list)):

        ckpt_path = os.path.join(args.weight_folder, ckpt)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint)
        model.train()

        enc_ckpt_path = os.path.join(args.weight_folder, enc_ckpt)
        enc_checkpoint = torch.load(enc_ckpt_path, map_location=device, weights_only=False)
        lq_encoder.load_state_dict(enc_checkpoint)
        lq_encoder.train()
        
        is_latent_data = True if "latent" in args.dataset else False

        cos_sim_t = {}
        ratio_t = {}

        for i, t_bin in enumerate(t_bins):
            model.zero_grad()
            fm_grad_accum = None
            cfm_grad_accum = None
            traj_grad_accum = None
            perc_grad_accum = None
            
            bar = tqdm(train_dataloader,disable=not accelerator.is_local_main_process)

            for iteration, (x_deg, x_gt) in enumerate(bar):
                bar.set_description(f"t_bin: {t_bin:.2f}, {(i) / len(t_bins) * 100:.2f}%")

                t = t_bin.repeat(batch_size).to(device)
                x_deg = x_deg.to(device, dtype=dtype, non_blocking=True)
                x_gt = x_gt.to(device, dtype=dtype, non_blocking=True)

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
                            z_cond = first_stage_model.encode(x_deg).latent_dist.sample().mul_(args.scale_factor)

                if args.conditional:
                    z_0 = torch.randn_like(z_1)
                else: 
                    z_0 = z_cond + args.sigma * torch.randn_like(z_cond)

                with accelerator.autocast():
                        
                    t = t.expand(z_1.size(0)).view(-1, 1, 1, 1)
                    t = (1 - args.consistency_dt) * t

                    if args.use_lpips_loss:
                        loss_type = 'cfm_elpips_loss'
                        loss_consistency, loss_trajectory, alpha, loss_velocity, loss_percep = cfm_lpips_loss(model, z_0, z_1, t, z_cond, elatentlpips_model, args)
                    elif args.use_lpl_loss:
                        loss_type = 'cfm_lpl_loss'
                        loss_consistency, loss_trajectory, alpha, loss_velocity, loss_percep, _ = cfm_lpl_loss(model, z_0, z_1, t, z_cond, args, decoder_extractor, percep_keys, percep_weights)
                    elif args.use_conflict_free:
                        loss_type = f'conflict_free_{args.projection_type}'

                        assert args.projection_type in ["one_projection", "two_projection"], \
                            "Please choose --projection_type: one_projection, two_projection"
                        if args.projection_type == 'one_projection':
                            assert args.one_projected_gradient in ["perceptual_gradient", "cfm_gradient"], \
                                "Please choose --one_projected_gradient: perceptual_gradient or cfm_gradient"
                            loss_type += f'_{args.one_projected_gradient}'

                        assert args.conflict_free_perceptual_type in ["elpips", "lpl"], \
                            "Please choose --conflict_free_perceptual_type: elpips, lpl"
                        if args.conflict_free_perceptual_type == 'elpips':
                            loss_type += f'_{args.one_projected_gradient}_{args.conflict_free_perceptual_type}'
                            loss_consistency, loss_trajectory, alpha, loss_velocity, loss_percep = cfm_lpips_loss(model, z_0, z_1, t, z_cond, elatentlpips_model, args)
                        elif args.conflict_free_perceptual_type == 'lpl':
                            loss_type += f'_{args.one_projected_gradient}_{args.conflict_free_perceptual_type}'
                            loss_consistency, loss_trajectory, alpha, loss_velocity, loss_percep, _ = cfm_lpl_loss(model, z_0, z_1, t, z_cond, args, decoder_extractor, percep_keys, percep_weights)

                        # perceptual loss weight
                        percep_weight = get_perceptual_weight(t, args)
                        loss_percep = (percep_weight * loss_percep).mean()
                    else:
                        loss_type = 'cfm_loss'
                        loss_consistency, loss_trajectory, alpha, loss_velocity = cfm_loss(model, z_0, z_1, t, z_cond, args)
                        loss_percep = None

                    m_grads = []

                    # LCPL
                    if not loss_percep == torch.zeros((), device=device):
                        model.zero_grad()
                        accelerator.backward(loss_percep, retain_graph=True)
                        m_grads.append(get_gradient_vector(model))
                    else:
                        perc_grad_accum = None

                    # LCFM
                    model.zero_grad()
                    loss_consistency = loss_trajectory + alpha * loss_velocity
                    accelerator.backward(loss_consistency.mean(), retain_graph=True)
                    m_grads.append(get_gradient_vector(model))
                    
                    # LCFM - trajectory loss 
                    model.zero_grad()
                    accelerator.backward(loss_trajectory, retain_graph=True)
                    m_grads.append(get_gradient_vector(model))

                    # LCFM - velocity loss
                    model.zero_grad()
                    loss_velocity = alpha * loss_velocity
                    accelerator.backward(loss_velocity.mean())
                    m_grads.append(get_gradient_vector(model))

                    grad_p, grad_cfm, grad_t, grad_v = m_grads

                    fm_grad_accum = grad_v if fm_grad_accum is None else fm_grad_accum + grad_v
                    cfm_grad_accum = grad_cfm if cfm_grad_accum is None else cfm_grad_accum + grad_cfm
                    traj_grad_accum = grad_t if traj_grad_accum is None else traj_grad_accum + grad_t
                    perc_grad_accum = grad_p if perc_grad_accum is None else perc_grad_accum + grad_p

            fm_grad_accum /= iteration + 1
            perc_grad_accum /= iteration + 1
            cfm_grad_accum /= iteration + 1
            traj_grad_accum /= iteration + 1

            grad_map = {
                    "fm": fm_grad_accum,
                    "cfm": cfm_grad_accum,
                    "traj": traj_grad_accum,
                    "perc": perc_grad_accum,
                }
            
            grad_map = {k: v for k, v in grad_map.items() if v is not None}

            for (name1, g1), (name2, g2) in itertools.combinations(grad_map.items(), 2):
                
                pair_name = f"{name1}_vs_{name2}"
                ratio_name = f"{name1}_over_{name2}"
                
                cos_sim_t.setdefault(pair_name, []).append(cosine_sim(g1, g2))
                ratio_t.setdefault(ratio_name, []).append(ratio(g1, g2))
        
        ratio_compare_objects = ratio_t.keys()
        cosine_compare_objects = cos_sim_t.keys()
        
        lambda_info = f"_lambda_{args.lambda_schedule}" if args.lambda_schedule != "constant" else ""

        # cosine similarity
        for c_object in cosine_compare_objects:
            c_numpy_title = f"{ckpt.removesuffix('.pth')}_{c_object}_{args.time_step_type}_{loss_type}{lambda_info}_cossim.npy"
            cosine_torch = torch.stack(cos_sim_t[c_object]).detach().cpu().numpy()
            if not args.debug:
                np.save(os.path.join(cosine_save_path, c_numpy_title), cosine_torch)
        # ratio
        for r_object in ratio_compare_objects:
            r_numpy_title = f"{ckpt.removesuffix('.pth')}_{r_object}_{args.time_step_type}_{loss_type}{lambda_info}_ratio.npy"
            ratio_torch = torch.stack(ratio_t[r_object]).detach().cpu().numpy()
            if not args.debug:
                np.save(os.path.join(ratio_save_path, r_numpy_title), ratio_torch)

    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pcflow gradient vector visualization")
    
    parser.add_argument("--seed", type=int, default=1024, help="seed used for initialization")
    parser.add_argument("--mode", type=str, default="train", help="train mode")

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
    parser.add_argument("--plot_every", type=int, default=5, help="plot every x epochs")

    # additional features (e.g., conditional/unconditional inputs, addition of noise, wandb)
    parser.add_argument('--conditional', action='store_true',
                    help='If set, the flow model is conditioned on either y or the posterior mean predictor. '
                            'Applies only to the stage "flow".')
    parser.add_argument("--sigma", type=float, default=0.0, help="gaussian noise std")

    # degradation type
    parser.add_argument('--degradation', type=str, required=True,
                        choices=['sr_bicubic_x8_gaussian_noise_005',
                                 'gaussian_noise_035',
                                 'colorization_gaussian_noise_025',
                                 'random_inpainting_gaussian_noise_01',
                                 'difface'],
                        help='The degradation type.')
    
    parser.add_argument("--first_stage_model_type", type=str, default="kl", choices=["kl", "taesd"], help="Type of first stage model")
    
    # perceptual loss parameters
    parser.add_argument("--use_lpl_loss", action='store_true', help="addition of perceptual loss")
    parser.add_argument("--lambda_percep", type=float, default=1.0, help="weight for perceptual loss")
    parser.add_argument("--percep_keys", nargs="+", type=str, default=["mid_block", "up_block_0", "up_block_1", "up_block_2", "up_block_3", "conv_out"])
    parser.add_argument("--percep_weights", nargs="+", type=float, default=[1, 1, 1, 1, 1, 1])

    parser.add_argument("--use_lpips_loss", action="store_true", help="Use E-Latent LPIPS as a distillation loss component")
    parser.add_argument("--lpips_normalization", type=bool, default=True, help="Activation for normalization")
    parser.add_argument("--taesd_lpips_pth",type=str, default=None,  help="root dir for latentLPIPS of taesd")
    parser.add_argument("--taesd_vgg_pth",type=str, default=None,  help="root dir for latentLPIPS of taesd")

    parser.add_argument("--method", type=str, default="euler", help="ODE solver")
    parser.add_argument("--steps", type=int, default=3, help="number of sampling steps")

    # gradient vector visualization
    parser.add_argument("--conflict_free_perceptual_type", type=str, default="constant", choices=["elpips", "lpl"], help="Choose perceptual type for conflict free")

    parser.add_argument("--time_step_type", type=str, default="t_bin", choices=["snr","t_bin"], help="mixed precision mode")
    parser.add_argument("--weight_folder",type=str, default=None,  help="weight_folder")
    parser.add_argument("--bin_numb",type=int, default=10,  help="bin_numb")
    parser.add_argument("--debug", action="store_true", help="debugging") # stop saving values
    parser.add_argument("--visualization_save_path",type=str, default=None,  help="save folder path for visualization save path")
    parser.add_argument("--train_len_control", action='store_true', help="activation train dataset number contorller")
    parser.add_argument("--train_length", type=int, default=70000, help="control number of train dataset")

    # consistency learning parameters
    parser.add_argument("--add_consistency_loss", action='store_true',
                        help="Add consistency flow matching loss to improve trajectory consistency")
    parser.add_argument("--consistency_alpha", type=float, default=0.001,
                        help="Weight for velocity consistency term within consistency loss (default: 0.001)")
    parser.add_argument("--consistency_k_steps", type=int, default=3,
                        help="Number of segments to divide the trajectory into (default: 5)")
    parser.add_argument("--consistency_dt", type=float, default=0.05,
                        help="Timestep delta for consistency computation (default: 0.05)")

    # lq_encoder
    parser.add_argument("--finetune_lq_encoder", action="store_true", help="Finetune a separate encoder for LQ images")

    # conflict-free gradient alignment
    parser.add_argument("--use_conflict_free",action="store_true",  help="use modified gradient vector loss")
    parser.add_argument("--projection_type", type=str, default="off", choices=["one_projection", "two_projection"], help="Choose projection type")
    parser.add_argument("--one_projected_gradient", type=str, default="off", choices=["perceptual_gradient", "cfm_gradient", "off"], help="Choose projected gradient")

    # scheduler
    parser.add_argument("--lambda_schedule", type=str, default="constant", choices=["constant", "in_linear", "in_linear_warmup","in_cosine", "de_linear", "de_cosine"], help="Choose lambda schedule type")

    args = parser.parse_args()
    cosine_similarity = calculate_cosine_similarity(args)