#!/bin/bash


export CUDA_VISIBLE_DEVICES=6  # set GPU ID here
# Limit CPU threads for PyTorch/BLAS
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# ========== CONFIG ==========
ANCHOR_ROOT="/datasets/vbr_slam"
MAPILLARY_ROOT="/home/bjangley/VPR/mapillary_utils/vbr_mapillary_overlap"
OUTPUT_ROOT="/home/bjangley/VPR/mast3r-v2/pairs_mapillary"
TOP_N=3
# ========== CONFIG ==========
SCENES=("spagna_train0" "campus_train0")

# ========== RUN ==========
for SCENE in "${SCENES[@]}"; do
  echo "[INFO] Processing $SCENE"
  python my_scripts/mine_mapillary.py \
    --anchor_scene "$SCENE" \
    --anchor_root "$ANCHOR_ROOT" \
    --query_root "$MAPILLARY_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --top_n "$TOP_N"
done
