#!/bin/bash

# --- User Configurable Paths ---
ROOT="/home/bjangley/VPR/vbr/spagna_train0_00"
CALIB_YAML="$ROOT/vbr_calib.yaml"
POSES="/home/bjangley/VPR/vbr/spagna_train0_gt.txt"
PAIRS_PATH="/home/bjangley/VPR/mast3r-v2/pairsVBR"
OUTPUT_DIR="$ROOT/depthmaps_npy"
CHECKPOINT="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"


# --- Dataset Arguments ---
TRAIN_DATASET="VBRPairsDataset(root_dir='$ROOT', split='train', pairs_txt='$PAIRS_PATH', calib_yaml='$CALIB_YAML', poses_txt='$POSES', depth_dir='$OUTPUT_DIR', resolution=[(512,384)], n_corres=1000, aug_crop=False)"
TEST_DATASET="VBRPairsDataset(root_dir='$ROOT', split='test', pairs_txt='$PAIRS_PATH', calib_yaml='$CALIB_YAML', poses_txt='$POSES', depth_dir='$OUTPUT_DIR', resolution=[(512,384)], n_corres=1000, aug_crop=False)"

# --- Loss Function ---
TRAIN_CRITERION="ConfLoss(Regr3D(L21, norm_mode='?avg_dis', gt_scale=True), alpha=0.2)"

# --- Model Arguments ---
MODEL="AsymmetricMASt3R(
    patch_embed_cls='PatchEmbedDust3R',
    img_size=(512, 512),
    enc_depth=24,
    dec_depth=12,
    enc_embed_dim=1024,
    dec_embed_dim=768,
    enc_num_heads=16,
    dec_num_heads=12,
    pos_embed='RoPE100',
    head_type='catmlp+dpt',
    output_mode='pts3d+desc24',
    depth_mode=('exp', -inf, inf),
    conf_mode=('exp', 1, inf),
    two_confs=True,
    desc_conf_mode=('exp', 0, inf),
    landscape_only=False
)"
##pretrained model path
PRETRAINED="/home/bjangley/VPR/mast3r-old/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"

# --- Training Hyperparameters ---
BATCH_SIZE=1
EPOCHS=10
ACCUM_ITER=4
LR=0.0001
MIN_LR=1e-06
WARMUP_EPOCHS=1
SAVE_FREQ=1
KEEP_FREQ=5
EVAL_FREQ=1
CUDA_VISIBLE_DEVICES=6
# --- Output Directory ---
OUTPUT_DIR_CHECKPOINTS="checkpoints/mast3r_vbr_demo"

# --- Launch Training ---
torchrun --nproc_per_node=1 --master_port=29501 train.py \
    --train_dataset "$TRAIN_DATASET" \
    --test_dataset "$TEST_DATASET" \
    --model "$MODEL" \
    --pretrained "$PRETRAINED" \
    --train_criterion "$TRAIN_CRITERION" \
    --test_criterion "$TRAIN_CRITERION" \
    --lr $LR \
    --min_lr $MIN_LR \
    --warmup_epochs $WARMUP_EPOCHS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --accum_iter $ACCUM_ITER \
    --save_freq $SAVE_FREQ \
    --keep_freq $KEEP_FREQ \
    --eval_freq $EVAL_FREQ \
    --disable_cudnn_benchmark \
    --output_dir "$OUTPUT_DIR_CHECKPOINTS"

  