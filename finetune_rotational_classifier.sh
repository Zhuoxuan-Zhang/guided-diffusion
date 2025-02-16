#!/bin/bash
# Set dataset path
DATASET_PATH="/users/zzhan513/data/zzhan513/visual_reasoning/train_repaint/rotational_train_set"
PRETRAINED_CHECKPOINT="models/256x256_classifier.pt"

# Reduce memory usage
# MODEL_FLAGS="--image_size 256 --num_channels 256 --num_res_blocks 2 --num_head_channels 64 --learn_sigma True --use_scale_shift_norm True --attention_resolutions 32,16,8 --class_cond True --resblock_updown True --use_fp16 True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --data_dir $DATASET_PATH --resume_checkpoint $PRETRAINED_CHECKPOINT"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_resblock_updown True --classifier_use_scale_shift_norm True"

python -m scripts.classifier_train --data_dir DATASET_PATH $TRAIN_FLAGS $CLASSIFIER_FLAGS