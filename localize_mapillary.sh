#!/bin/bash

export CUDA_VISIBLE_DEVICES=6
# Limit CPU threads for PyTorch/BLAS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

DATASET_ROOT="/datasets/vbr_slam"
MAPILLARY_ROOT="/home/bjangley/VPR/mapillary_utils/vbr_mapillary_downloads"
PAIRS_ROOT="/home/bjangley/VPR/mast3r-v2/pairs_mapillary"   
OUTPUT_ROOT="/home/bjangley/VPR/mast3r-v2/results_mapillary"
TEMP_PREFIX="${OUTPUT_ROOT}/temp"
CHECKPOINT="/home/bjangley/VPR/mast3r-v2/checkpoints_v0/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

CONF_PERCENTILE=0
MIN_INLIERS=200

SCENES=(
  "spagna_train0"
  "campus_train0"
  "campus_train1"
)

echo "---------------- CONFIG ----------------"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "DATASET_ROOT         = $DATASET_ROOT"
echo "MAPILLARY_ROOT       = $MAPILLARY_ROOT"
echo "PAIRS_ROOT           = $PAIRS_ROOT"         
echo "OUTPUT_ROOT          = $OUTPUT_ROOT"
echo "TEMP_PREFIX          = $TEMP_PREFIX"
echo "CHECKPOINT           = $CHECKPOINT"
echo "CONF_PERCENTILE      = $CONF_PERCENTILE"
echo "MIN_INLIERS          = $MIN_INLIERS"
echo "SCENES               = ${SCENES[*]}"
echo "----------------------------------------"

for SCENE in "${SCENES[@]}"; do
  echo "=== Running $SCENE ==="

  python my_scripts/localize_mapillary.py \
    --dataset_scenes "$SCENE" \
    --dataset_root "$DATASET_ROOT" \
    --mapillary_root "$MAPILLARY_ROOT" \
    --pairs_root "$PAIRS_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --model_path "$CHECKPOINT" \
    --conf_percentile "$CONF_PERCENTILE" \
    --min_inliers "$MIN_INLIERS" \
    --temp_file_prefix "$TEMP_PREFIX"\

  echo "=== Done: $SCENE ==="
done
