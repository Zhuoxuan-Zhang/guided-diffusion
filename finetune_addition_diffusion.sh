#!/bin/bash
# Set dataset path
DATASET_PATH="/users/zzhan513/data/zzhan513/visual_reasoning/mnist_training_imgs/addition_training_imgs/gt/"
PRETRAINED_CHECKPOINT="models/256x256_diffusion.pt"

# Reduce memory usage
MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 2 --num_head_channels 64 --learn_sigma True --use_scale_shift_norm True --attention_resolutions 32,16,8 --class_cond True --resblock_updown True --use_fp16 True"
DIFFUSION_FLAGS="--diffusion_steps 200 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --use_fp16 True --data_dir $DATASET_PATH --resume_checkpoint $PRETRAINED_CHECKPOINT"

mpiexec -n 2 python -m scripts.image_train $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
