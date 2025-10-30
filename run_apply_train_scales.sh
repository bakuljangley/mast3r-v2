#!/bin/bash

# Define common parameters
DATASET_ROOT="/datasets/vbr_slam"
PAIRS_PATH="pairs_finetuning/"  #root directory to the pairs
SPLIT="all"
OUTPUT_ROOT="results_localization/original_train_scale/"
CHECKPOINT="/home/bjangley/VPR/mast3r-v2/checkpoints_v0/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
# CHECKPOINT="/home/bjangley/VPR/mast3r-v2/checkpoints_v1/ciampino1_ciampino2_lr5e5_3e6_reg/checkpoint-best.pth"
TEMP_PREFIX="results_localization/original_train_scale/temp" # Prefix for temp files
SCALE_JSON="my_vbr_utils/train_scales.json"   # <-- replace with your precomputed scales JSON

export CUDA_VISIBLE_DEVICES=5
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# Scenes to process (space-separated list)
SCENES="ciampino_train0"

# Run the evaluation script
python my_scripts/apply_train_scales.py \
  --dataset_scenes $SCENES \
  --dataset_root "$DATASET_ROOT" \
  --pairs_path "$PAIRS_PATH" \
  --split "$SPLIT" \
  --conf_percentile 0\
  --output_root "$OUTPUT_ROOT" \
  --model_path "$CHECKPOINT" \
  --min_inliers 200 \
  --temp_file_prefix "$TEMP_PREFIX" \
  --scale_json "$SCALE_JSON"
