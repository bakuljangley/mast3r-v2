#!/bin/bash

DATASET_SCENE="campus_train1"
DATASET_ROOT="/datasets/vbr_slam" 
PAIRS_FILE="/home/bjangley/VPR/mast3r-v2/pairs_mining/campus_train1/campus_matches_inliers_fm_top10_anchors_per_query.csv" 
OUTPUT_DIR="/home/bjangley/VPR/mast3r-v2/pairs_finetuning/"    
DEPTH_OUTPUT_DIR="/home/bjangley/VPR/vbr/depths" 
POSES_OUTPUT_DIR="/home/bjangley/VPR/vbr/poses"  
MIN_INLIERS=700
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
