import argparse
import os
import json
import glob
import numpy as np
from time import time

import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from diffusers.models import AutoencoderKL

from models import create_network, DecoderFeatureExtractor, TAESDWrapper, TAESDDecoderFeatureExtractor
from datasets_prep import get_ir_dataset
from utils.create_degradation import create_degradation
from torchdiffeq import odeint

from lpips import LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from pytorch_fid.fid_score import calculate_fid_given_paths

from utils import compute_niqe, compute_musiq


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

def load_train_config(dataset, model_type, exp):
    """
    Load training configuration.
    ./saved_info/pcflow/{dataset}/{model_type}/{exp}/config.yaml
    """
    exp_path = os.path.join("./saved_info/pcflow", dataset, model_type, exp)
    cfg_path = os.path.join(exp_path, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    return cfg, exp_path

def select_real_img_dir(args):
    """Load pre-calculated real image statistics for FID calculation."""
    if args.precomputed_statistics:
        if args.dataset == "cifar10":
            real_img_dir = "pytorch_fid/cifar10_train_stat.npy"
        elif args.dataset == "celeba_256":
            real_img_dir = "pytorch_fid/celebahq_stat.npy"
        elif args.dataset == "lsun_church":
            real_img_dir = "pytorch_fid/lsun_church_stat.npy"
        elif args.dataset == "ffhq_256":
            real_img_dir = "pytorch_fid/ffhq_stat.npy"
        elif args.dataset == "lsun_bedroom":
            real_img_dir = "pytorch_fid/lsun_bedroom_stat.npy"
        elif args.dataset in ["latent_imagenet_256", "imagenet_256"]:
            real_img_dir = "pytorch_fid/imagenet_stat.npy"
        else:
            real_img_dir = args.real_img_dir
    else:
        real_img_dir = args.real_img_dir
    return real_img_dir

def save_metrics_to_json(metrics: dict, filepath: str):
    """
    Save metrics to a JSON file.
    
    Args:
        metrics (dict): Dictionary containing metric names and their corresponding values.
        filepath (str): JSON file path to save the metrics.
    """
    try:
        data = []
        data.append(metrics)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)

    except Exception as e:
        raise(f"ERROR: Could not save metrics to JSON: {e}")


def main(args):
    accelerator = Accelerator()
    device = accelerator.device
    dtype = torch.float32
    set_seed(args.seed + accelerator.process_index)

    train_cfg, exp_path = load_train_config(args.dataset, args.model_type, args.exp)
    train_args = argparse.Namespace(**OmegaConf.to_object(train_cfg))
 
    args.degradation = train_args.degradation # degradation type identical to training 
    test_dataset = get_ir_dataset(args)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    model = create_network(train_args).to(device, dtype=dtype)
    if train_args.use_grad_checkpointing and "DiT" in train_args.model_type:
        model.set_gradient_checkpointing()

    if train_args.first_stage_model_type == "kl":
        first_stage_model = AutoencoderKL.from_pretrained(train_args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)     
    elif train_args.first_stage_model_type == "taesd":
        first_stage_model = TAESDWrapper(train_args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)    
    else:
        raise ValueError(f"Unknown first_stage_model_type: {train_args.first_stage_model_type}")

    first_stage_model.eval()
    for p in first_stage_model.parameters():
        p.requires_grad = False

    # load finetuned LQ encoder if available
    lq_encoder = None
    if getattr(train_args, 'finetune_lq_encoder', False):
        if train_args.first_stage_model_type == "kl":
            lq_encoder = AutoencoderKL.from_pretrained(train_args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)
        elif train_args.first_stage_model_type == "taesd":
            lq_encoder = TAESDWrapper(train_args.pretrained_autoencoder_ckpt).to(device, dtype=dtype)

        model_ckpt_name = os.path.splitext(args.model_ckpt)[0]
        epoch_num = model_ckpt_name.split("_")[-1]
        lq_encoder_ckpt_path = os.path.join(exp_path, f"lq_encoder_{epoch_num}.pth")
        if os.path.exists(lq_encoder_ckpt_path):
            lq_encoder_state = torch.load(lq_encoder_ckpt_path, map_location="cpu")
            lq_encoder.load_state_dict(lq_encoder_state, strict=True)
        else:
            lq_encoder = None

        if lq_encoder is not None:
            lq_encoder.eval()
            for p in lq_encoder.parameters():
                p.requires_grad = False

    # load model checkpoint
    ckpt_path = os.path.join(exp_path, args.model_ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    test_loader, model = accelerator.prepare(test_loader, model)

    # remove warning messages 
    import warnings
    warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
    warnings.filterwarnings("ignore", message="Arguments other than a weight enum")
    warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

    # initialize metrics
    lpips_metric = LPIPS(net="vgg").to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    psnr_metric.reset()
    ssim_metric.reset()
    lpips_values = []

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(
            exp_path,
            f"test_{args.test_dataset}_{os.path.splitext(args.model_ckpt)[0]}_{args.method}_steps_{args.steps}",
        )
    
    save_dir_img = f'{save_dir}/img_files_{epoch_num}epoch'
    
    if accelerator.is_main_process and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    if accelerator.is_main_process and not os.path.exists(save_dir_img):
        os.makedirs(save_dir_img , exist_ok=True)

    world_size = accelerator.num_processes
    rank = accelerator.process_index

    # ----- inference
    model.eval()
    first_stage_model.eval()
    start_time = time()

    with torch.no_grad():
        for batch_idx, (x_deg, x_gt) in enumerate(test_loader):
            x_deg = x_deg.to(device, dtype=dtype)

            if lq_encoder is not None:
                z_cond = lq_encoder.encode(x_deg).latent_dist.sample().mul_(train_args.scale_factor)
            else:
                z_cond = first_stage_model.encode(x_deg).latent_dist.sample().mul_(train_args.scale_factor)

            if train_args.conditional:
                rand = torch.randn_like(z_cond)
                z_1_pred = sample_from_model(model, rand, z_cond=z_cond, steps=args.steps, method=args.method)[-1]
            else:
                z_source = z_cond + train_args.sigma * torch.randn_like(z_cond)
                z_1_pred = sample_from_model(model, z_source, steps=args.steps, method=args.method)[-1]

            x_pred = first_stage_model.decode(z_1_pred / train_args.scale_factor).sample  # [-1,1]
            x_pred = torch.clamp(x_pred, min=0.0, max=1.0) # clamping to [0,1]
            
            # PSNR / SSIM: [0,1]
            psnr_metric.update(x_pred, x_gt)
            ssim_metric.update(x_pred, x_gt)

            # LPIPS: [-1,1]
            lpips_val = lpips_metric(x_pred * 2 - 1, x_gt * 2 - 1)
            lpips_values.append(lpips_val.mean().item())

            x_pred_cpu = torch.clamp(x_pred.detach().cpu(), min=0.0, max=1.0)
            b = x_pred_cpu.size(0)
            base_idx = batch_idx * args.batch_size * world_size

            for j in range(b):
                idx = base_idx + j * world_size + rank
                save_path = os.path.join(save_dir_img, f"{idx:06d}.png")
                torchvision.utils.save_image(x_pred_cpu[j], save_path)

            if accelerator.is_main_process and batch_idx % 20 == 0:
                elapsed = time() - start_time
                accelerator.print(
                    f"[{batch_idx}/{len(test_loader)}] "
                    f"saved up to idx {idx}, {elapsed:.1f}s"
                )
                start_time = time()

    psnr_value = float(psnr_metric.compute().item())
    ssim_value = float(ssim_metric.compute().item())
    lpips_value = float(sum(lpips_values) / len(lpips_values))

    if accelerator.is_main_process:
        restored_paths = sorted(glob.glob(os.path.join(save_dir_img, "*.png")))
        if len(restored_paths) == 0:
            raise RuntimeError(f"No restored images found in: {save_dir_img}")

        real_img_dir = select_real_img_dir(args)
        if args.precomputed_statistics:
            from pytorch_fid.fid_score import (
                InceptionV3 as FID_InceptionV3,
                compute_statistics_of_path,
                calculate_frechet_distance,
            )
            stat = np.load(real_img_dir, allow_pickle=True).item()
            m2, s2 = stat['mu'], stat['sigma']
            block_idx = FID_InceptionV3.BLOCK_INDEX_BY_DIM[2048]
            fid_inception = FID_InceptionV3([block_idx]).to(device)
            m1, s1 = compute_statistics_of_path(
                save_dir_img, fid_inception, 200, 2048, device
            )
            fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        else:
            paths = [save_dir_img, real_img_dir]
            fid_value = calculate_fid_given_paths(
                paths=paths, batch_size=200, device=device, dims=2048
            )

        metrics_result = {
            "dataset_test": args.test_dataset,
            "dataset_real": args.dataset,
            "FID": fid_value,
            "PSNR": psnr_value,
            "SSIM": ssim_value,
            "LPIPS": lpips_value,
        }
        log_str = (
            f"[{args.test_dataset}] "
            f"FID={fid_value:.2f} | PSNR={psnr_value:.2f} | SSIM={ssim_value:.3f} | "
            f"LPIPS={lpips_value:.3f}"
        )

        if args.niqe_musiq_activation:
            niqe_dict = compute_niqe(restored_paths, device=str(device))
            musiq_dict = compute_musiq(restored_paths, device=str(device))
            metrics_result.update({**niqe_dict, **musiq_dict})
            log_str += f" | NIQE={niqe_dict['NIQE']:.3f} | MUSIQ={musiq_dict['MUSIQ']:.3f}"

        accelerator.print(log_str)

        metric_folderpath = os.path.join(save_dir,'json_result')
        os.makedirs(metric_folderpath, exist_ok=True)
        metric_filepath = os.path.join(metric_folderpath, f"metrics_summary_{epoch_num}epoch.json")
        save_metrics_to_json(metrics_result, metric_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("pcflow inference parameters")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--mode", type=str, default="test", help="test mode")

    # training experiment info
    parser.add_argument("--exp", type=str, required=True, help="experiment name")
    parser.add_argument("--image_size", type=int, default=512, help="size of image")
    parser.add_argument("--dataset", type=str, required=True, help="real dataset name for FID stats (e.g., ffhq_256, celeba_256)")

    # test dataset
    parser.add_argument("--test_dataset", type=str, required=True, help="name of IR test dataset (e.g., ffhq_256, celeba_test)")
    parser.add_argument("--test_deg_datadir", type=str, default=None, help="root dir for test dataset (degraded image)")
    parser.add_argument("--test_gt_datadir", type=str, default=None, help="root dir for test dataset (gt image)")

    # sampling
    parser.add_argument("--model_ckpt", type=str, required=True, help="checkpoint file name relative to exp dir (e.g., model_1200.pth)")
    parser.add_argument("--model_type", type=str, help="pre-trained model type")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default=None, help="where to save reconstructed images (for FID). default: under exp folder")
    parser.add_argument("--steps", type=int, default=3, help="number of sampling steps")
    parser.add_argument("--method", type=str, default="euler", help="ODE solver")

    # compute metrics
    parser.add_argument("--precomputed_statistics", action="store_true", help="activation for precalculated statistics for computing FID. if not set, use real_img_dir")
    parser.add_argument("--real_img_dir", type=str, default="", help="optional override path for real stats/images. if empty, choose from args.dataset")
    parser.add_argument("--niqe_musiq_activation", action="store_true", help="activation for calculating niqe and musiq")

    args = parser.parse_args()
    main(args)

