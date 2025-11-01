#!/bin/bash

### --- CONFIG --- ###
DATASET_ROOT="/datasets/vbr_slam"
DATASET_SCENE="ciampino_train0"
TOP_N=5

#output directories
PAIRS_ROOT="/pairs_mining/"
OUTPUT_DIR="/pairs_finetuning/"
OUTPUT_ROOT="/vbr"
### --- CONFIG --- ###

DEPTH_OUTPUT_DIR="${OUTPUT_ROOT}/depths"
POSES_OUTPUT_DIR="${OUTPUT_ROOT}/poses"
PAIRS_FILE="${PAIRS_ROOT}${DATASET_SCENE}/matches_inliers_fm_top${TOP_N}_anchors_per_query.csv"
MIN_INLIERS_VALID=200      # valid if >200
HIGH_INLIERS_THRESH=700    # high if >700
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
