#!/bin/bash

# Define common parameters
DATASET_ROOT="/datasets/vbr_slam"
PAIRS_PATH="pairs_finetuning/"  # Directory containing <scene>_pairs.txt
SPLIT="all"
OUTPUT_ROOT="results_localization/original/"
CHECKPOINT="/home/bjangley/VPR/mast3r-v2/checkpoints_v0/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
TEMP_PREFIX="results_localization/original/temp_depthpro" # Prefix for temp files
export CUDA_VISIBLE_DEVICES=4

# Scenes to process (space-separated list)
SCENES="spagna_train0 campus_train0 campus_train1 ciampino_train0 ciampino_train1"

# Run the evaluation script
python my_scripts/evaluate_anydepth.py \
  --dataset_scenes $SCENES \
  --dataset_root "$DATASET_ROOT" \
  --pairs_path "$PAIRS_PATH" \
  --split "$SPLIT" \
  --conf_percentile 0\
  --output_root "$OUTPUT_ROOT" \
  --model_path "$CHECKPOINT" \
  --min_inliers 200 \
  --temp_file_prefix "$TEMP_PREFIX"