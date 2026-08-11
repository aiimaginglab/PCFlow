#!/bin/bash
BASE_PATH="/path/to/pcflow"
DATA_DIR="/path/to/pcflow/data"

# # ===== unconditional ===== #
# CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision "bf16" --num_processes 1 \
#   $BASE_PATH/train_flow_latent_internal_percep.py \
#       --exp BS128_BF16_NF128_ffhq_lunet_taesd_sr_adm_ema_sigma_0_05_consistency_0_to_1 \
#       --dataset ffhq_256 --train_gt_datadir $DATA_DIR/FFHQ/images1024x1024 --valid_gt_datadir $DATA_DIR/celeba_test/celeba_512_validation \
#       --model_type lunet --first_stage_model_type taesd --pretrained_autoencoder_ckpt madebyollin/taesd3 \
#       --batch_size 128 --num_epoch 250 \
#       --image_size 256 --f 8 --num_in_channels 16 --num_out_channels 16 \
#       --nf 128 --ch_mult 1 2 1 2 --n_mid_blocks 1 \
#       --lr 2e-4 --beta1 0.9 --beta2 0.999 --use_ema --ema_decay 0.999 --weight_decay 0.02 \
#       --scale_factor 1.0 \
#       --save_ckpt_every 50 \
#       --sigma 0.05 \
#       --mixed_precision bf16 \
#       --degradation 'sr_bicubic_x8_gaussian_noise_005' \
#       --add_consistency_loss \
#       --consistency_alpha 0.001 \
#       --consistency_k_steps 3 \
#       --consistency_dt 0.05 \
#       --wandb_logging \

# ===== Conditional ===== #
CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision "bf16" --num_processes 1 \
  $BASE_PATH/train_flow_latent_internal_percep.py \
      --exp BS128_BF16_NF128_ffhq_lunet_taesd_sr_adm_ema_sigma_0_05_cond_finetune_consistency_0_to_1 \
      --dataset ffhq_256 --train_gt_datadir $DATA_DIR/FFHQ/images1024x1024 --valid_gt_datadir $DATA_DIR/celeba_test/celeba_512_validation \
      --model_type lunet --first_stage_model_type taesd --pretrained_autoencoder_ckpt madebyollin/taesd3 \
      --batch_size 128 --num_epoch 250 \
      --image_size 256 --f 8 --num_in_channels 32 --num_out_channels 16 \
      --nf 128 --ch_mult 1 2 1 2 --n_mid_blocks 1 \
      --lr 2e-4 --beta1 0.9 --beta2 0.999 --use_ema --ema_decay 0.999 --weight_decay 0.02 \
      --scale_factor 1.0 \
      --save_ckpt_every 50 \
      --sigma 0.05 \
      --mixed_precision bf16 \
      --degradation 'sr_bicubic_x8_gaussian_noise_005' \
      --add_consistency_loss \
      --consistency_alpha 0.001 \
      --consistency_k_steps 3 \
      --consistency_dt 0.05 \
      --conditional \
      --finetune_lq_encoder \
      --method "euler" \
      --wandb_logging \
