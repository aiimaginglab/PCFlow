#!/bin/bash
BASE_PATH="/path/to/pcflow"
DATA_DIR="/path/to/pcflow/data"
EXP="sr" # "sr" # "denoising" # "inpainting" # "colorization" # "bfr"
MODEL_TYPE="lfm-fr-consistency-loss" # "lfm-fr-consistency-loss" / "lfm-fr-lpl-loss" 
DEVICE=0

############################################### Super Resolution/Denoising/Inpainting/Colorization ###############################################

CUDA_VISIBLE_DEVICES=$DEVICE \
accelerate launch --num_processes 1 $BASE_PATH/test_flow_latent_internal_percep.py \
  --exp $EXP \
  --image_size 256 \
  --dataset ffhq_256 \
  --test_dataset celeba_test \
  --test_gt_datadir $DATA_DIR/celeba_test/celeba_512_validation \
  --model_ckpt model_450.pth \
  --model_type $MODEL_TYPE \
  --batch_size 64 \
  --steps 3 \
  --method "euler" \
  --precomputed_statistics

############################################### BFR ###############################################

CUDA_VISIBLE_DEVICES=$DEVICE \
accelerate launch --num_processes 1 $BASE_PATH/test_flow_latent_internal_percep.py \
  --exp $EXP \
  --image_size 512 \
  --dataset ffhq_256 \
  --test_dataset celeba_test \
  --test_gt_datadir $DATA_DIR/celeba_test/celeba_512_validation \
  --test_deg_datadir $DATA_DIR/celeba_test/celeba_512_validation_lq \
  --real_img_dir $DATA_DIR/FFHQ/images512x512 \
  --model_ckpt model_500.pth \
  --model_type $MODEL_TYPE \
  --batch_size 32 \
  --steps 5 \
  --method "euler" \
  --niqe_musiq_activation