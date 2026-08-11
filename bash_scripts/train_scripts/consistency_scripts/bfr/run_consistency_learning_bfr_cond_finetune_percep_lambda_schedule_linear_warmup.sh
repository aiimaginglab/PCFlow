#!/bin/bash
BASE_PATH="/path/to/pcflow"
DATA_DIR="/path/to/pcflow/data"
PRETRAINED_BASE_PATH="/path/to/pretrained_models"
PRETRAINED_MSE_DIR=$PRETRAINED_BASE_PATH/saved_info/pcflow/ffhq_256/lfm-fr-consistency-loss
PRETRAINED_MSE_EXP="BS128_BF16_NF128_ffhq_lunet_taesd_bfr_adm_ema_sigma_0_05_cond_finetune_consistency_0_to_1"

echo "================================ 🎨 BFR ============================================="

debugging=true # true / false
CUDA_VISIBLE_DEVICES=0

task=bfr

# lambda manipulation (perceptual weight)
lambda_percep=1
lambda_schedule=in_linear_warmup # constant # in_linear # in_linear_warmup

# alpha 
consistency_alpha=0.001 

# loss manipulation
loss_type="use_conflict_free" # "use_lpips_loss" # "use_lpl_loss" # "use_conflict_free"
projection_type="one_projection" # when your loss type is conflict free, choose "one_projection" or "two_projection"
one_projected_gradient="cfm_gradient" # when you choose one_projection, you must choose "perceptual_gradient" or "cfm_gradient"

if [ "$debugging" = true ]; then wandb_flag=""; echo "############################ ⚠️ NoW DeBuGgInG MoD ⚠️ ############################"
else wandb_flag='--wandb_logging'
fi

loss_flag="--${loss_type}"

if [ "$loss_type" = "use_lpips_loss" ]; then exp_set="_elpips"
elif [ "$loss_type" = "use_lpl_loss" ]; then exp_set="_lpl"
elif [ "$loss_type" = "use_conflict_free" ]; then exp_set="_conflict_free_${projection_type}"
  if [ "$projection_type" = "one_projection" ]; then  exp_set="_conflict_free_${projection_type}_${one_projected_gradient}"
  elif [ "$projection_type" = "two_projection" ]; then  exp_set="_conflict_free_${projection_type}"
  fi
else loss_flag="" exp_set=""
fi


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --mixed_precision "bf16" --num_processes 1 \
  $BASE_PATH/train_flow_latent_internal_percep.py \
      --exp "${task}/BS128_BF16_NF128_ffhq_lunet_taesd_${task}_adm_ema_sigma_0_05_cond_finetune_consistency_0_to_1_lambda_pecep_${lambda_percep}_${lambda_schedule}_lpl_${exp_set}" \
      --dataset ffhq_256 --train_gt_datadir $DATA_DIR/FFHQ/images1024x1024 --valid_gt_datadir $DATA_DIR/celeba_test/celeba_512_validation \
      --model_type lunet --first_stage_model_type taesd --pretrained_autoencoder_ckpt madebyollin/taesd3 \
      --batch_size 32 --num_epoch 250 \
      --image_size 512 --f 8 --num_in_channels 32 --num_out_channels 16 \
      --nf 128 --ch_mult 1 2 1 2 --n_mid_blocks 3 \
      --lr 2e-4 --beta1 0.9 --beta2 0.999 --use_ema --ema_decay 0.999 --weight_decay 0.02 \
      --scale_factor 1.0 \
      --save_ckpt_every 50 \
      --sigma 0.05 \
      --mixed_precision bf16 \
      --degradation 'difface' \
      --add_consistency_loss \
      --consistency_alpha $consistency_alpha \
      --consistency_k_steps 5 \
      --consistency_dt 0.05 \
      --conditional \
      --finetune_lq_encoder \
      --method "euler" \
      --use_lpl_loss --lambda_percep $lambda_percep \
      --percep_keys mid_block up_block_0 up_block_1 up_block_2 conv_out \
      --percep_weights 1.0 0.5 0.25 0.125 1 \
      --model_ckpt $PRETRAINED_MSE_DIR/$PRETRAINED_MSE_EXP/model_250.pth \
      --lq_encoder_ckpt $PRETRAINED_MSE_DIR/$PRETRAINED_MSE_EXP/lq_encoder_250.pth \
      --projection_type $projection_type \
      --one_projected_gradient $one_projected_gradient \
      --lambda_schedule $lambda_schedule \
      $loss_flag \
      $wandb_flag \
