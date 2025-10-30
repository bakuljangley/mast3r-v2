#!/bin/bash

# Define common parameters
DATASET_ROOT="/datasets/vbr_slam"
PAIRS_PATH="pairs_finetuning/"  #root directory to the pairs
SPLIT="all"
OUTPUT_ROOT="results_localization/original_conf40/"
CHECKPOINT="/home/bjangley/VPR/mast3r-v2/checkpoints_v0/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
# CHECKPOINT="/home/bjangley/VPR/mast3r-v2/checkpoints_v1/ciampino1_ciampino2_lr5e5_3e6_reg/checkpoint-best.pth"
TEMP_PREFIX="results_localization/original_conf40/temp" # Prefix for temp files
export CUDA_VISIBLE_DEVICES=5
# Limit CPU threads for PyTorch/BLAS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# Scenes to process (space-separated list)
SCENES="ciampino_train0"

# Run the evaluation script
python my_scripts/evaluate_v5.py \
  --dataset_scenes $SCENES \
  --dataset_root "$DATASET_ROOT" \
  --pairs_path "$PAIRS_PATH" \
  --split "$SPLIT" \
  --conf_percentile 40\
  --output_root "$OUTPUT_ROOT" \
  --model_path "$CHECKPOINT" \
  --min_inliers 200 \
  --temp_file_prefix "$TEMP_PREFIX"