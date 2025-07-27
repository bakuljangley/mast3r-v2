#!/bin/bash
SCENE="spagna_train0" #to evaluate on
DATASET_ROOT="/datasets/vbr_slam"
PAIRS_PATH="/home/bjangley/VPR/mast3r-v2/pairs_mining/spagna_train0/spagna_matches_inliers_fm_top10_anchors_per_query.csv"
OUTPUT_FOLDER="results_localization/spagna_train0/spagna"
TEMP="results_localization/spagna_train0/spagna_processed_pairs.txt" 

export CUDA_VISIBLE_DEVICES=4

python my_scripts/evaluate_v3.py \
  --dataset_scene "$SCENE" \
  --dataset_root "$DATASET_ROOT" \
  --pairs_csv "$PAIRS_PATH" \
  --output_prefix "$OUTPUT_FOLDER" \
  --model_name "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric" \
  --min_inliers 200 \
  --temp_file "$TEMP"