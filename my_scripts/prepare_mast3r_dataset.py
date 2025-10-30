import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import random
import numpy as np
from PIL import Image
from pathlib import Path
from my_vbr_utils.vbr_dataset import vbrInterpolatedDataset, get_paths_from_scene, load_calibration
from my_utils.my_vbr_dataset import generate_depth_and_scene_maps
from my_utils.transformations import pose_to_se3, se3_to_pose
import argparse
import logging

# --- Configuration ---
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare VPR dataset splits with inlier thresholds.")
    parser.add_argument("--dataset_scene", type=str, required=True, help="Name of the scene.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory to dataset.")
    parser.add_argument("--pairs_file", type=str, required=True, help="Path to the pairs CSV file.")
    # Thresholds per spec: valid > 200; high > 700
    parser.add_argument("--min_inliers_valid", type=int, default=200, help="Pairs with num_inliers > this are considered valid.")
    parser.add_argument("--high_inliers_thresh", type=int, default=700, help="High-inlier threshold for train/val/test1 creation.")
    # Ratios apply ONLY to the high-inlier pool
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio (over high-inlier pool).")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Val ratio (over high-inlier pool).")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test1 ratio (over high-inlier pool).")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for splits.")
    parser.add_argument("--depth_output_dir", type=str, required=True, help="Path to save depth maps.")
    parser.add_argument("--poses_output_dir", type=str, required=True, help="Path to save poses.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for deterministic splits.")
    return parser.parse_args()

# --- Helpers ---
def load_and_bucket_pairs(pairs_file, min_inliers_valid=200, high_thresh=700):
    """
    Read all pairs from CSV, then:
      - valid_df: num_inliers > min_inliers_valid
      - high_df:  num_inliers > high_thresh  (subset of valid)
      - mid_df:   min_inliers_valid < num_inliers <= high_thresh
    Returns dict with dataframes and pair lists (anchor_idx, query_idx).
    """
    pairs_df = pd.read_csv(pairs_file)
    total = len(pairs_df)

    valid_df = pairs_df[pairs_df['num_inliers'] > min_inliers_valid].copy()
    high_df  = valid_df[valid_df['num_inliers'] > high_thresh].copy()
    mid_df   = valid_df[valid_df['num_inliers'] <= high_thresh].copy()  # still > min_inliers_valid

    def to_pairs(df):
        return list(zip(df['anchor_idx'].astype(int), df['query_idx'].astype(int)))

    valid_pairs = to_pairs(valid_df)
    high_pairs  = to_pairs(high_df)
    mid_pairs   = to_pairs(mid_df)

    logging.info(f"Total pairs in CSV: {total}")
    logging.info(f"Valid   (> {min_inliers_valid}): {len(valid_df)}")
    logging.info(f"High    (> {high_thresh}): {len(high_df)}")
    logging.info(f"Mid     ({min_inliers_valid}..{high_thresh}] : {len(mid_df)}")

    return {
        "pairs_df": pairs_df,
        "valid_df": valid_df,
        "high_df": high_df,
        "mid_df": mid_df,
        "valid_pairs": valid_pairs,
        "high_pairs": high_pairs,
        "mid_pairs": mid_pairs
    }

def split_high_train_val_test(high_pairs, train_ratio, val_ratio, test_ratio, random_seed=42):
    random.seed(random_seed)
    high_pairs = high_pairs[:]   
    random.shuffle(high_pairs)

    n = len(high_pairs)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = int(n * test_ratio)

    train_pairs = high_pairs[:n_train]
    val_pairs   = high_pairs[n_train:n_train + n_val]
    test1_pairs = high_pairs[n_train + n_val:n_train + n_val + n_test]

    logging.info(f"[High split] train:{len(train_pairs)} val:{len(val_pairs)} test1:{len(test1_pairs)} from {n}")
    return train_pairs, val_pairs, test1_pairs


def save_pairs_to_file(pairs, output_file):
    """Save pairs as 'anchor_idx query_idx' per line."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for a, q in pairs:
                f.write(f"{a} {q}\n")
        logging.info(f"Pairs saved to {output_file} ({len(pairs)} lines)")
    except Exception as e:
        logging.error(f"Error saving pairs to {output_file}: {e}")

def generate_depth_maps(pairs, vbr_scene, calib, output_dir):
    """
    Generate and save depth maps for all unique image indices referenced by the provided pairs.
    """
    T_cam_lidar = calib['cam_l']["T_cam_lidar"]
    K = calib['cam_l']['K']
    os.makedirs(output_dir, exist_ok=True)

    # Unique image indices
    image_indices = sorted(set([idx for pair in pairs for idx in pair]))

    for image in image_indices:
        try:
            item = vbr_scene[image]
            img_path = item['image']
            lidar_pts = item['lidar_points']
            if lidar_pts.shape[0] < 5:
                logging.warning(f"[{image}] Skipped (insufficient lidar points)")
                continue
            img = Image.open(img_path)
            img_shape = img.size[::-1]  # (H, W)
            depth, scene = generate_depth_and_scene_maps(lidar_pts, K, T_cam_lidar, img_shape)
            out_path = os.path.join(output_dir, f"{image:010d}.npy")
            np.save(out_path, depth.astype(np.float32))
        except Exception as e:
            logging.error(f"Error generating depth map for image {image}: {e}")

def save_poses(vbr_scene, output_file, T_cam_lidar):
    """
    Save ground-truth poses (as provided by vbr_scene) to a text file.
    """
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for idx in range(len(vbr_scene)):
                pose = vbr_scene.get_pose(idx)
                # If camera-frame needed instead, use:
                # pose_cam = se3_to_pose(T_cam_lidar @ pose_to_se3(pose))
                f.write(f"{pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f} {pose[3]:.6f} {pose[4]:.6f} {pose[5]:.6f} {pose[6]:.6f}\n")
    except Exception as e:
        logging.error(f"Error saving poses to {output_file}: {e}")

# --- Main ---
if __name__ == "__main__":
    args = parse_args()

    # Logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Dataset
    logging.info(f"Loading dataset for scene: {args.dataset_scene}")
    vbr_scene = vbrInterpolatedDataset(args.dataset_root, args.dataset_scene)
    calib_path = get_paths_from_scene(args.dataset_root, args.dataset_scene)[-1]
    calib = load_calibration(calib_path)
    T_cam_lidar = calib['cam_l']["T_cam_lidar"]

    # Load and bucket pairs
    buckets = load_and_bucket_pairs(
        args.pairs_file,
        min_inliers_valid=args.min_inliers_valid,
        high_thresh=args.high_inliers_thresh
    )
    valid_pairs = buckets["valid_pairs"]
    high_pairs  = buckets["high_pairs"]
    mid_pairs   = buckets["mid_pairs"]

    if len(high_pairs) == 0:
        logging.warning("No pairs with inliers > high_inliers_thresh to form train/val/test1. Exiting.")
        sys.exit(0)

    # Split ONLY high-inlier pool into train/val/test1
    train_pairs, val_pairs, test1_pairs = split_high_train_val_test(
        high_pairs,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        random_seed=args.random_seed
    )

    # test2 = test1 ∪ mid (mid: valid but not >700). Disjoint by construction, so concat.
    test2_pairs = test1_pairs + mid_pairs
    logging.info(f"[test2] size: {len(test2_pairs)} (test1: {len(test1_pairs)} + mid: {len(mid_pairs)})")

    # Save splits
    scene_pairs_path = os.path.join(args.output_dir, args.dataset_scene)
    os.makedirs(scene_pairs_path, exist_ok=True)

    save_pairs_to_file(train_pairs, os.path.join(scene_pairs_path, "train_pairs.txt"))
    save_pairs_to_file(val_pairs,   os.path.join(scene_pairs_path, "val_pairs.txt"))
    save_pairs_to_file(test1_pairs, os.path.join(scene_pairs_path, "test1_pairs.txt"))
    save_pairs_to_file(test2_pairs, os.path.join(scene_pairs_path, "test2_pairs.txt"))

    # Optional: all valid pairs (>200) for reference
    save_pairs_to_file(valid_pairs, os.path.join(scene_pairs_path, "all_pairs.txt"))

    # Depth maps for all valid pairs (>200)
    depth_output_dir = os.path.join(args.depth_output_dir, args.dataset_scene)
    logging.info("Generating depth maps for valid pairs (> min_inliers_valid)")
    generate_depth_maps(valid_pairs, vbr_scene, calib, depth_output_dir)

    # Save poses
    poses_output_file = os.path.join(args.poses_output_dir, f"{args.dataset_scene}.txt")
    logging.info("Saving poses")
    save_poses(vbr_scene, poses_output_file, T_cam_lidar)

    logging.info("Dataset preparation complete.")
