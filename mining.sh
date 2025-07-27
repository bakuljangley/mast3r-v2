#!/bin/bash
SCENE="ciampino_train0" #to evaluate on
DATASET_ROOT="/datasets/vbr_slam"
SEQUENCE_PATH="/home/bjangley/VPR/mast3r-v2/my_vbr_utils/vbr_sequences/ciampino_train0.json"
OUTPUT="pairs_mining/ciampino_train0/ciampino_matches_inliers_fm.csv"
TEMP="pairs_mining/ciampino_train0/processed_pairs.txt" 

export CUDA_VISIBLE_DEVICES=4

python my_scripts/mining.py \
  --dataset_scene "$SCENE" \
  --dataset_root "$DATASET_ROOT" \
  --anchor_query_json "$SEQUENCE_PATH" \
  --anchor_step 10 \
  --query_step 20 \
  --output "$OUTPUT" \
  --top_n 3 \
  --temp_file "$TEMP"

