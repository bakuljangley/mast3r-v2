#!/bin/bash

DATASET_SCENE="ciampino_train0"
TOP_N=5
MIN_INLIERS_VALID=200      # valid if >200
HIGH_INLIERS_THRESH=700    # high if >700

# ROOT DIRECTORIES
DATASET_ROOT="/datasets/vbr_slam"
PAIRS_ROOT="/home/bjangley/VPR/mast3r-v2/pairs_mining/"
OUTPUT_DIR="/home/bjangley/VPR/mast3r-v2/pairs_finetuning/"
DEPTH_OUTPUT_DIR="/home/bjangley/VPR/vbr/depths"
POSES_OUTPUT_DIR="/home/bjangley/VPR/vbr/poses"

DATASET_NAME=$(echo "$DATASET_SCENE" | cut -d'_' -f1) # Extract part before first underscore
PAIRS_FILE="${PAIRS_ROOT}${DATASET_SCENE}/${DATASET_NAME}_matches_inliers_fm_top${TOP_N}_anchors_per_query.csv"

TRAIN_RATIO=0.7
VAL_RATIO=0.15
TEST_RATIO=0.15

# --- Run the Python script with arguments ---
python my_scripts/prepare_mast3r_dataset.py \
  --dataset_scene "$DATASET_SCENE" \
  --dataset_root "$DATASET_ROOT" \
  --pairs_file "$PAIRS_FILE" \
  --min_inliers_valid "$MIN_INLIERS_VALID" \
  --high_inliers_thresh "$HIGH_INLIERS_THRESH" \
  --train_ratio "$TRAIN_RATIO" \
  --val_ratio "$VAL_RATIO" \
  --test_ratio "$TEST_RATIO" \
  --output_dir "$OUTPUT_DIR" \
  --depth_output_dir "$DEPTH_OUTPUT_DIR" \
  --poses_output_dir "$POSES_OUTPUT_DIR"
