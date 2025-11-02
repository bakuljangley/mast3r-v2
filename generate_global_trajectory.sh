#!/bin/bash

# SCENES=("campus_train0" "campus_train1" "ciampino_train0" "ciampino_train1" "spagna_train0")
SCENES=("spagna_train0")
###--- CONFIG ---###
DATASET_ROOT="/datasets/vbr_slam"
UTILS_ROOT="/home/bjangley/VPR/mast3r-v2/my_vbr_utils/"
SAVE_ROOT="/home/bjangley/VPR/mast3r-v2/my_vbr_utils/global_trajectory/"
###--- CONFIG ---###
for SCENE in "${SCENES[@]}"; do
  echo "Processing scene: $SCENE"
  python my_scripts/generate_global_trajectory.py \
    --scene "$SCENE" \
    --dataset_root "$DATASET_ROOT" \
    --utils_root "$UTILS_ROOT" \
    --save_dir "$SAVE_ROOT"
done
