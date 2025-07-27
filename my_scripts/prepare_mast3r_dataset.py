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
# Moved configuration parameters to argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Prepare VPR dataset splits.")
    parser.add_argument("--dataset_scene", type=str, required=True, help="Name of the scene.")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory to dataset.")
    parser.add_argument("--pairs_file", type=str, required=True, help="Path to the pairs CSV file.")
    parser.add_argument("--min_inliers", type=int, default=200, help="Minimum number of inliers for filtering.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of pairs for training.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of pairs for validation.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of pairs for testing.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--depth_output_dir", type=str, required=True, help="Path to save depth maps.")
    parser.add_argument("--poses_output_dir", type=str, required=True, help="Path to save poses.")
    return parser.parse_args()

# --- Helper Functions ---

def filter_pairs(pairs_file, min_inliers):
    """Filters pairs based on a minimum number of inliers."""
    try:
        pairs_df = pd.read_csv(pairs_file)
        filtered_rows = pairs_df[pairs_df['num_inliers'] > min_inliers]
        anchor_idxs = filtered_rows['anchor_idx'].astype(int).tolist()
        query_idxs = filtered_rows['query_idx'].astype(int).tolist()
        pairs = [(anchor_idxs[idx], query_idxs[idx]) for idx in range(len(anchor_idxs))]
        logging.info(f"Total pairs before filtering: {len(pairs_df)}")
        logging.info(f"Total pairs after filtering: {len(filtered_rows)}")
        return pairs
    except FileNotFoundError:
        logging.error(f"Pairs file not found: {pairs_file}")
        return None

def split_pairs(pairs, train_ratio, val_ratio, test_ratio, random_seed=42):
    """Splits pairs into training, validation, and test sets."""
    random.seed(random_seed)
    random.shuffle(pairs)
    total_pairs = len(pairs)
    train_end = int(total_pairs * train_ratio)
    val_end = train_end + int(total_pairs * val_ratio)
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]
    logging.info(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}, Test pairs: {len(test_pairs)}")
    return train_pairs, val_pairs, test_pairs

def save_pairs_to_file(pairs, output_file):
    """Saves pairs to a text file."""
    try:
        with open(output_file, "w") as f:
            for pair in pairs:
                f.write(f"{pair[0]} {pair[1]}\n")
        logging.info(f"Pairs saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving pairs to {output_file}: {e}")

def generate_depth_maps(pairs, vbr_scene, calib, output_dir):
    """Generates and saves depth maps for the given pairs."""
    T_cam_lidar = calib['cam_l']["T_cam_lidar"]
    K = calib['cam_l']['K']
    os.makedirs(output_dir, exist_ok=True)
    for idx in range(len(pairs)):
        pair = pairs[idx]
        for image in pair:
            try:
                item = vbr_scene[image]
                img_path = item['image']
                lidar_pts = item['lidar_points']
                if lidar_pts.shape[0] < 5:
                    logging.warning(f"[{image}] Skipped (no lidar)")
                    continue
                img = Image.open(img_path)
                img_shape = img.size[::-1]  # (H, W)
                depth, scene = generate_depth_and_scene_maps(lidar_pts, K, T_cam_lidar, img_shape)
                out_path = os.path.join(output_dir, f"{image:010d}.npy")
                np.save(out_path, depth.astype(np.float32))
                # logging.info(f"[{image}] Saved to {out_path}")
            except Exception as e:
                logging.error(f"Error generating depth map for image {image}: {e}")

def save_poses(vbr_scene, output_file, T_cam_lidar):
    """Saves ground truth poses to a text file."""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for idx in range(len(vbr_scene)):
                # Ensure the output directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                pose = vbr_scene.get_pose(idx)
                pose_cam = se3_to_pose(T_cam_lidar @ pose_to_se3(pose))
                f.write(f"{pose_cam[0]:.6f} {pose_cam[1]:.6f} {pose_cam[2]:.6f} {pose_cam[3]:.6f} {pose_cam[4]:.6f} {pose_cam[5]:.6f} {pose_cam[6]:.6f}\n")
        # logging.info(f"Ground truth poses saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving poses to {output_file}: {e}")

# --- Main Script ---
if __name__ == "__main__":
    args = parse_args()

    # --- Logging setup ---
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Dataset Loading ---
    logging.info(f"Loading dataset for scene: {args.dataset_scene}")
    vbr_scene = vbrInterpolatedDataset(args.dataset_root, args.dataset_scene)
    calib_path = get_paths_from_scene(args.dataset_root, args.dataset_scene)[-1]
    calib = load_calibration(calib_path)
    T_cam_lidar = calib['cam_l']["T_cam_lidar"]

    # --- Pair Splitting ---
    logging.info("Splitting pairs into train, val, and test sets.")
    pairs = filter_pairs(args.pairs_file, args.min_inliers)
    if pairs:
        train_pairs, val_pairs, test_pairs = split_pairs(pairs, args.train_ratio, args.val_ratio, args.test_ratio)

        # --- Saving Pairs ---
        scene_pairs_path = os.path.join(args.output_dir, args.dataset_scene)
        os.makedirs(scene_pairs_path, exist_ok=True)
        save_pairs_to_file(train_pairs, os.path.join(scene_pairs_path, "train_pairs.txt"))
        save_pairs_to_file(val_pairs, os.path.join(scene_pairs_path, "val_pairs.txt"))
        save_pairs_to_file(test_pairs, os.path.join(scene_pairs_path, "test_pairs.txt"))
        save_pairs_to_file(pairs, os.path.join(scene_pairs_path, "all_pairs.txt"))

        # --- Generate and Save Depth Maps ---
        depth_output_dir = os.path.join(args.depth_output_dir, args.dataset_scene)
        logging.info("Generating depth maps")
        generate_depth_maps(pairs, vbr_scene, calib, depth_output_dir)

        # --- Save Poses ---
        poses_output_file = os.path.join(args.poses_output_dir, f"{args.dataset_scene}.txt")
        logging.info("Saving poses")
        save_poses(vbr_scene, poses_output_file, T_cam_lidar)

        logging.info("Dataset preparation complete.")
    else:
        logging.warning("No pairs found after filtering.")