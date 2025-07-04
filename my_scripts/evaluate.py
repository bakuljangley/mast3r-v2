import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from my_utils.my_vbr_dataset import vbrDataset, load_calibration, generate_depth_and_scene_maps
from my_utils.mast3r_utils import (
    get_master_output, get_mast3r_image_shape, scale_intrinsics, overlap,
    pose_to_se3, se3_to_pose, solve_pnp, quaternion_rotational_error
)
from my_utils.scaling import scale_pnp, compute_scaled_points

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate poses from anchor-query pairs CSV.")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--calib', type=str, required=True)
    parser.add_argument('--pairs_csv', type=str, required=True, help='CSV with anchor_idx,query_idx')
    parser.add_argument('--output_prefix', type=str, default='estimates')
    parser.add_argument('--device', type=str, default='cuda:4')
    parser.add_argument('--model_name', type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument('--min_inliers', type=int, default=200, help='Minimum inliers required for a match to be valid')
    parser.add_argument('--temp_file', type=str, required=True, help='File to log pairs that have already been processed')
    return parser.parse_args()


def compute_statistics(pose, anchor_pose, mast3r_pts=None, lidar_pts=None):
    """
    Compute statistics for pose estimation.
    
    Args:
        pose: Estimated pose [tx, ty, tz, qx, qy, qz, qw].
        anchor_pose: Ground truth pose [tx, ty, tz, qx, qy, qz, qw].
        mast3r_pts: 3D points from MASt3R (optional).
        lidar_pts: 3D points from LiDAR (optional).
    
    Returns:
        pos_error: Total positional error.
        x_error, y_error, z_error: Positional errors along x, y, z axes.
        rot_error: Rotational error (angular distance between quaternions).
        pointmap_error: Error between MASt3R and LiDAR point clouds (optional).
    """
    # Compute positional errors
    pos_error = np.linalg.norm(np.array(pose[:3]) - np.array(anchor_pose[:3]))
    x_error, y_error, z_error = np.abs(np.array(pose[:3]) - np.array(anchor_pose[:3]))
    
    # Compute rotational error using quaternion_rotational_error
    rot_error = quaternion_rotational_error(np.array(pose[3:]), np.array(anchor_pose[3:]))
    
    # Compute pointmap error if 3D points are provided
    pointmap_error = np.linalg.norm(mast3r_pts - lidar_pts) if mast3r_pts is not None and lidar_pts is not None else None
    
    return pos_error, x_error, y_error, z_error, rot_error, pointmap_error

def save_estimation_and_statistics(output_file, query_idx, anchor_idx, pose, statistics, fmt="%.6f"):
    """
    Append a single pose and statistics row to the output files.
    """
    pose_row = [query_idx, anchor_idx] + list(pose)
    with open(output_file, "a") as f:
        np.savetxt(f, np.array([pose_row]), fmt=fmt)
    stats_file = output_file.replace(".txt", "_statistics.txt")
    stats_row = [query_idx, anchor_idx] + list(statistics)
    with open(stats_file, "a") as f:
        np.savetxt(f, np.array([stats_row]), fmt=fmt)

def load_processed_pairs(filename):
    if not os.path.exists(filename):
        return set()
    with open(filename, "r") as f:
        return set(tuple(map(int, line.strip().split(','))) for line in f)

def mark_pair_processed(filename, anchor_idx, query_idx):
    with open(filename, "a") as f:
        f.write(f"{anchor_idx},{query_idx}\n")

def main():
    args = parse_args()

    # LOAD DATASET AND CALIBRATION FILE 
    dataset = vbrDataset(args.dataset, args.gt)
    calib = load_calibration(args.calib)
    K = calib['cam_l']['K']
    T_base_cam = calib['cam_l']['T_base_cam']
    T_cam_lidar = calib['cam_l']['T_cam_lidar']

    # LOAD MAST3R MODEL AND INFERENCE UTILITIES
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)

    # Load anchor-query pairs
    pairs_df = pd.read_csv(args.pairs_csv)
    processed_pairs = load_processed_pairs(args.temp_file)
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Estimating poses"):
        anchor_idx, query_idx = int(row['anchor_idx']), int(row['query_idx'])
        if (anchor_idx, query_idx) in processed_pairs:
            continue  # skip already processed
        anchor = dataset[anchor_idx]
        query = dataset[query_idx]

        # MASt3R matching
        output = get_master_output(model, args.device,
            anchor['image_left'], query['image_left'],
            visualize=False, verbose=False
        )
        matches_im0 = output[0]
        matches_im1 = output[1]
        pts3d_im0   = output[2]

        # Inlier filtering via fundamental matrix
        if len(matches_im0) >= 8:
            F, mask_f = cv2.findFundamentalMat(matches_im0, matches_im1, cv2.FM_RANSAC, 0.5, 0.99)
            inlier_mask = mask_f.ravel().astype(bool)
            inlier_im0 = matches_im0[inlier_mask]
            inlier_im1 = matches_im1[inlier_mask]
        else:
            inlier_im0, inlier_im1 = np.empty((0, 2)), np.empty((0, 2))

        n_matches = len(matches_im0)
        n_inliers = len(inlier_im0)

        if len(inlier_im0) > args.min_inliers:
            # LiDAR depth + intrinsics rescaling
            img = cv2.imread(anchor['image_left'])
            H, W = img.shape[:2]
            mast3r_w, mast3r_h = get_mast3r_image_shape(W, H)
            K_new = scale_intrinsics(K, W, H, mast3r_w, mast3r_h)
            depth_map, scene_map = generate_depth_and_scene_maps(anchor['lidar_points'], K_new, T_cam_lidar, (mast3r_h, mast3r_w))

            # Overlap matching
            valid_mast3r_uv, valid_lidar_uv, matched_idx = overlap(inlier_im0, depth_map, max_pixel_dist=2)
            np.array(matched_idx, dtype=int)
            if len(matched_idx) > 0:
                inlier_im0 = inlier_im0[matched_idx]
                inlier_im1 = inlier_im1[matched_idx]
            else:
                inlier_im0 = np.empty((0, 2))
                inlier_im1 = np.empty((0, 2))
            mast3r_pts = pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]]
            lidar_pts = scene_map[valid_lidar_uv[:, 1], valid_lidar_uv[:, 0]]

            T_anchor_base = pose_to_se3(anchor['pose'])

            # MASt3R estimation
            if len(inlier_im0) >= 4:
                T_query_anchor_mast3r = solve_pnp(
                    mast3r_pts,
                    inlier_im1, K_new
                )
                if T_query_anchor_mast3r is not None:
                    T_m_local = T_anchor_base @ T_base_cam @ np.linalg.inv(T_query_anchor_mast3r)
                    pose_m = se3_to_pose(T_m_local)

                    # Compute statistics
                    pos_error, x_error, y_error, z_error, rot_error, _ = compute_statistics(
                        pose_m, anchor['pose']
                    )
                    median_depth_mast3r = np.median(mast3r_pts[:, 2]) if mast3r_pts.size > 0 else None
                    mast3r_statistics = [n_matches, n_inliers, median_depth_mast3r, x_error, y_error, z_error, pos_error, rot_error]

                    # Save immediately
                    save_estimation_and_statistics(
                        f"{args.output_prefix}_mast3r.txt",
                        query_idx, anchor_idx, pose_m, mast3r_statistics
                    )

            # LiDAR estimation
            if len(valid_lidar_uv) >= 4:
                T_query_anchor_lidar = solve_pnp(
                    lidar_pts,
                    inlier_im1, K_new
                )
                if T_query_anchor_lidar is not None:
                    T_l_local = T_anchor_base @ T_base_cam @ np.linalg.inv(T_query_anchor_lidar)
                    pose_l = se3_to_pose(T_l_local)

                    # Compute statistics
                    pos_error, x_error, y_error, z_error, rot_error, _ = compute_statistics(
                        pose_l, anchor['pose']
                    )
                    median_depth_lidar = np.median(lidar_pts[:, 2]) if lidar_pts.size > 0 else None
                    lidar_statistics = [query_idx, anchor_idx, n_matches, n_inliers, median_depth_lidar, x_error, y_error, z_error, pos_error, rot_error]

                    # Save immediately
                    save_estimation_and_statistics(
                        f"{args.output_prefix}_lidar.txt",
                        query_idx, anchor_idx, pose_l, lidar_statistics
                    )

            # Scaled estimations (v3, v4, ICP)
            for scale_type, output_file in [
                ('v3', f"{args.output_prefix}_mast3r_scaled_v3.txt"),
                ('v4', f"{args.output_prefix}_mast3r_scaled_v4.txt"),
                ('icp', f"{args.output_prefix}_mast3r_scaled_icp.txt")
            ]:
                scaled_mast3r_pc, _ = compute_scaled_points(scale_type, mast3r_pts, lidar_pts)
                T_scaled = scale_pnp(scale_type, mast3r_pts, lidar_pts, inlier_im1, K_new)
                if T_scaled is not None:
                    T_scaled_local = T_anchor_base @ T_base_cam @ np.linalg.inv(T_scaled)
                    pose_scaled = se3_to_pose(T_scaled_local)

                    # Compute statistics
                    pos_error, x_error, y_error, z_error, rot_error, pointmap_error = compute_statistics(
                        pose_scaled, anchor['pose'], mast3r_pts, lidar_pts
                    )
                    median_depth = np.median(scaled_mast3r_pc[:, 2]) if scaled_mast3r_pc.size > 0 else None
                    statistics = [query_idx, anchor_idx, n_matches, n_inliers, median_depth, x_error, y_error, z_error, pos_error, rot_error, pointmap_error]

                    # Save immediately
                    save_estimation_and_statistics(
                        output_file,
                        query_idx, anchor_idx, pose_scaled, statistics
                    )
        # Mark as processed only after all processing and saving
        mark_pair_processed(args.temp_file, anchor_idx, query_idx)

if __name__ == "__main__":
    main()
