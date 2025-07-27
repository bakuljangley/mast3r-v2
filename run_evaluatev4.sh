#!/bin/bash
SCENE="campus_train1" #to evaluate on
DATASET_ROOT="/datasets/vbr_slam"
PAIRS_PATH="pairs_finetuning/campus_train1/all_pairs.txt"
OUTPUT_FOLDER="results_finetuning/spagna_train0_campus_train0/campus_train1"
TEMP="results_finetuning/spagna_train0_campus_train0/campus_train1_processed_pairs.txt"
CHECKPOINT="/home/bjangley/VPR/mast3r-v2/checkpoints/spagna_train0_campus_train0/checkpoint-best.pth"
# CHECKPOINT="/home/bjangley/VPR/mast3r-old/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
export CUDA_VISIBLE_DEVICES=4

python my_scripts/evaluate_v4.py \
--dataset_scene "$SCENE" \
--dataset_root "$DATASET_ROOT" \
--pairs_path "$PAIRS_PATH" \
--output_prefix "$OUTPUT_FOLDER" \
--model_path "$CHECKPOINT" \
--min_inliers 200 \
--temp_file "$TEMP"