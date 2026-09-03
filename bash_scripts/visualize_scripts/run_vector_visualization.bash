#!/bin/bash
BASE_PATH="/path/to/pcflow"
DATA_DIR="/path/to/pcflow/data"

debugging=true # true / false
CUDA_VISIBLE_DEVICES=0

task=sr # sr, inpainting, colorization, denoising

# lambda manipulation (perceptual weight)
lambda_percep=1
lambda_schedule="in_linear_warmup" # constant # in_linear # in_linear_warmup

# alpha 
consistency_alpha=0.001 

# loss manipulation
loss_type="use_lpl_loss" # "use_lpips_loss" # "use_lpl_loss" # "use_conflict_free"
projection_type="one_projection" # when your loss type is conflict free, choose "one_projection" or "two_projection"
one_projected_gradient="cfm_gradient" # when you choose one_projection, you must choose "perceptual_gradient" or "cfm_gradient"

BIN_number=30
train_control_act=true
train_length=1280 
batch_size=128

conflict_free_perceptual_type="lpl"

weight_folder="path/to/weights"
visualization_save_path=$BASE_PATH/visualization_analysis/$task/

if [ "$train_control_act" = true ]; then
    train_len_control="--train_len_control"
else
    train_len_control=""
fi

if [ "$debugging" = true ]; then 
    
    train_length=10
    BIN_number=10 
    batch_size=10
    debug='--debug' # do not save values
    visualization_save_path=$BASE_PATH/visualization_analysis/debug/$task
    echo "############################ ⚠️ NoW DeBuGgInG MoD ⚠️ ############################"
else 
    debug=''
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

if [ "$task" = "sr" ]; then degradation="sr_bicubic_x8_gaussian_noise_005"
elif [ "$task" = "inpainting" ]; then degradation="random_inpainting_gaussian_noise_01"
elif [ "$task" = "denoising" ]; then degradation="gaussian_noise_035"
elif [ "$task" = "colorization" ]; then degradation="colorization_gaussian_noise_025"
else 
  echo "ValueError: choose task [sr, inpainting, denoising, colorization]" >&2
  exit 1
fi

echo "                                 Exp Setting                           "
echo "################ Task: $task"
echo "################ Loss: $loss_type"
echo "################ BIN_number: $BIN_number"
echo "################ Batch_size: $batch_size"
echo "################ Train_length: $train_length" 


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES accelerate launch --mixed_precision "bf16" --num_processes 1 $BASE_PATH/vector_visualization.py \
    --weight_folder $weight_folder \
    --visualization_save_path $visualization_save_path \
    --dataset ffhq_256 --train_gt_datadir $DATA_DIR/FFHQ/images1024x1024 --valid_gt_datadir $DATA_DIR/celeba_test/celeba_512_validation \
    --model_type lunet --first_stage_model_type taesd --pretrained_autoencoder_ckpt madebyollin/taesd3 \
    --batch_size $batch_size --num_epoch 450 \
    --image_size 256 --f 8 --num_in_channels 32 --num_out_channels 16 \
    --nf 128 --ch_mult 1 2 1 2 --n_mid_block 1 \
    --lr 2e-4 --beta1 0.9 --beta2 0.999 --use_ema --ema_decay 0.999 --weight_decay 0.02 \
    --scale_factor 1.0 \
    --save_ckpt_every 50 \
    --sigma 0.05 \
    --mixed_precision bf16 \
    --degradation $degradation \
    --add_consistency_loss \
    --consistency_alpha $consistency_alpha  \
    --consistency_k_steps 3 \
    --consistency_dt 0.05 \
    --conditional \
    --finetune_lq_encoder \
    --method "euler" \
    $loss_flag --lambda_percep $lambda_percep \
    --percep_keys mid_block up_block_0 up_block_1 up_block_2 conv_out \
    --percep_weights 1.0 0.5 0.25 0.125 1 \
    --projection_type $projection_type \
    --one_projected_gradient $one_projected_gradient \
    --lambda_schedule $lambda_schedule \
    --conflict_free_perceptual_type $conflict_free_perceptual_type \
    $train_len_control \
    --bin_numb $BIN_number \
    --train_length $train_length \
    --time_step_type "t_bin" \
    $debug \