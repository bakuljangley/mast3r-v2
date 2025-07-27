#!/bin/bash

DATASET_SCENE="ciampino_train0"
DATASET_ROOT="/datasets/vbr_slam" 
PAIRS_FILE="/home/bjangley/VPR/mast3r-v2/pairs_mining/ciampino_train0/ciampino_matches_inliers_fm_top3_anchors_per_query.csv" 
OUTPUT_DIR="/home/bjangley/VPR/mast3r-v2/pairs_finetuning/ciampino_train0/"    
DEPTH_OUTPUT_DIR="/home/bjangley/VPR/vbr_final/depths" 
POSES_OUTPUT_DIR="/home/bjangley/VPR/vbr_final/poses"  
MIN_INLIERS=200
TRAIN_RATIO=0.7
VAL_RATIO=0.15
TEST_RATIO=0.15


# --- Run the Python script with arguments ---
python my_scripts/prepare_mast3r_dataset.py \
  --dataset_scene "$DATASET_SCENE" \
  --dataset_root "$DATASET_ROOT" \
  --pairs_file "$PAIRS_FILE" \
  --min_inliers "$MIN_INLIERS" \
  --train_ratio "$TRAIN_RATIO" \
  --val_ratio "$VAL_RATIO" \
  --test_ratio "$TEST_RATIO" \
  --output_dir "$OUTPUT_DIR" \
  --depth_output_dir "$DEPTH_OUTPUT_DIR" \
  --poses_output_dir "$POSES_OUTPUT_DIR"
