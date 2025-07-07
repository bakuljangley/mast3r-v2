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

def ensure_dir_exists(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

def compute_statistics(pose, anchor_pose, mast3r_pts=None, lidar_pts=None):
    pos_error = np.linalg.norm(np.array(pose[:3]) - np.array(anchor_pose[:3]))
    x_error, y_error, z_error = np.abs(np.array(pose[:3]) - np.array(anchor_pose[:3]))
    rot_error = quaternion_rotational_error(np.array(pose[3:]), np.array(anchor_pose[3:]))
    if mast3r_pts is not None and lidar_pts is not None and len(mast3r_pts) == len(lidar_pts):
        pointmap_error = np.mean(np.linalg.norm(mast3r_pts - lidar_pts, axis=1))
    else:
        pointmap_error = np.nan
    return pos_error, x_error, y_error, z_error, rot_error, pointmap_error

def save_result(output_file, anchor_seq, query_seq, query_idx, anchor_idx, pose, statistics, status="OK", fmt="%.6f"):
    """
    Save the pose and statistics to the output file.
    """
    ensure_dir_exists(output_file)
    pose_row = [query_idx, anchor_idx] + list(pose)
    with open(output_file, "a") as f:
        np.savetxt(f, np.array([pose_row]), fmt=fmt)
    
    stats_file = output_file.replace(".txt", "_statistics.txt")
    stats_row = [query_seq, anchor_seq, query_idx, anchor_idx] + list(statistics) + [status]
    

    # Adjust format specifier to match the number of elements in stats_row
    stats_fmt = " ".join(["%.6f"] * (len(stats_row)-1) + ["%s"])  # Add "%s" for the status string
    with open(stats_file, "a") as f:
        np.savetxt(f, np.array([stats_row], dtype=object), fmt=stats_fmt)


def save_failed_result(output_file, anchor_seq, query_seq, query_idx, anchor_idx, n_matches, n_inliers, median_depth, stat_len):
    pose = [np.nan] * 7
    statistics = [n_matches, n_inliers, median_depth] + [np.nan] * (stat_len - 3)
    save_result(output_file, anchor_seq, query_seq, query_idx, anchor_idx, pose, statistics, status="FAILED")

def load_processed_pairs(filename):
    if not os.path.exists(filename):
        return set()
    with open(filename, "r") as f:
        return set(tuple(map(int, line.strip().split(','))) for line in f)

def mark_pair_processed(filename, anchor_idx, query_idx):
    ensure_dir_exists(filename)
    with open(filename, "a") as f:
        f.write(f"{anchor_idx},{query_idx}\n")

def get_matches(model, device, anchor_img, query_img):
    output = get_master_output(model, device, anchor_img, query_img, visualize=False, verbose=False)
    return output[0], output[1], output[2]

def filter_inliers(matches_im0, matches_im1):
    if len(matches_im0) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    F, mask_f = cv2.findFundamentalMat(matches_im0, matches_im1, cv2.FM_RANSAC, 0.5, 0.99)
    inlier_mask = mask_f.ravel().astype(bool)
    return matches_im0[inlier_mask], matches_im1[inlier_mask]

def get_intrinsics_and_maps(anchor, K, T_cam_lidar):
    img = cv2.imread(anchor['image_left'])
    H, W = img.shape[:2]
    mast3r_w, mast3r_h = get_mast3r_image_shape(W, H)
    K_new = scale_intrinsics(K, W, H, mast3r_w, mast3r_h)
    depth_map, scene_map = generate_depth_and_scene_maps(anchor['lidar_points'], K_new, T_cam_lidar, (mast3r_h, mast3r_w))
    return K_new, depth_map, scene_map

def get_overlap_points(inlier_im0, depth_map):
    valid_mast3r_uv, valid_lidar_uv, matched_idx = overlap(inlier_im0, depth_map, max_pixel_dist=2)
    if len(matched_idx) > 0:
        return matched_idx, valid_lidar_uv
    return np.empty((0, 2)), np.empty((0, 2))

def process_pair(model, anchor, query, anchor_seq, query_seq, anchor_idx, query_idx, K, T_base_cam, T_cam_lidar, args):
    try:    
        matches_im0, matches_im1, pts3d_im0 = get_matches(model, args.device, anchor['image_left'], query['image_left'])
        inlier_im0, inlier_im1 = filter_inliers(matches_im0, matches_im1)
        n_matches, n_inliers = len(matches_im0), len(inlier_im0)

        # Always save, even if failed (so all files align)
        method_specs = [
            ('mast3r', 10, f"{args.output_prefix}_mast3r.txt"),
            ('lidar', 10, f"{args.output_prefix}_lidar.txt"),
            ('v3', 12, f"{args.output_prefix}_mast3r_scaled_v3.txt"),
            ('v4', 14, f"{args.output_prefix}_mast3r_scaled_v4.txt"),
            ('icp', 12, f"{args.output_prefix}_mast3r_scaled_icp.txt"),
        ]

        if n_inliers <= args.min_inliers or inlier_im0.size == 0:
            for method, stat_len, output_file in method_specs:
                save_failed_result(output_file, anchor_seq, query_seq, query_idx, anchor_idx, n_matches, n_inliers, np.nan, stat_len)
            return

        K_new, depth_map, scene_map = get_intrinsics_and_maps(anchor, K, T_cam_lidar)
        matched_idx, valid_lidar_uv = get_overlap_points(inlier_im0, depth_map)
        
        # Track number of overlapping points
        n_overlapping = len(matched_idx)

        if inlier_im0.size == 0 or matched_idx.size==0:
            for method, stat_len, output_file in method_specs:
                save_failed_result(output_file, anchor_seq, query_seq, query_idx, anchor_idx, n_matches, n_inliers, np.nan, stat_len)
            return
        inlier_im0 = inlier_im0[matched_idx]
        inlier_im1 = inlier_im1[matched_idx]
        mast3r_pts = pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]]
        lidar_pts = scene_map[valid_lidar_uv[:, 1], valid_lidar_uv[:, 0]]

        if len(inlier_im0) < 4:
            for method, stat_len, output_file in method_specs:
                save_failed_result(output_file, anchor_seq, query_seq, query_idx, anchor_idx, n_matches, n_inliers, np.nan, stat_len)
            return

        # Compute ground truth pose
        T_local_anchor = pose_to_se3(anchor['pose'])
        T_local_query = pose_to_se3(query['pose'])
        T_query_anchor_gt = np.linalg.inv(T_local_query @ T_base_cam) @ T_local_anchor @ T_base_cam
        pose_gt = se3_to_pose(T_query_anchor_gt)

        # Compute distance between anchor and query
        anchor_position = np.array(anchor['pose'][:3])  # Extract position (x, y, z) from anchor pose
        query_position = np.array(query['pose'][:3])    # Extract position (x, y, z) from query pose
        distance_anchor_query = np.linalg.norm(anchor_position - query_position)  # Euclidean distance
        # --- Estimation methods ---
        for method, stat_len, output_file in method_specs:
            if method == 'mast3r':
                T = solve_pnp(mast3r_pts, inlier_im1, K_new)
            elif method == 'lidar':
                T = solve_pnp(lidar_pts, inlier_im1, K_new)
            else:
                scaled_pts, scale, T = scale_pnp(method, mast3r_pts, lidar_pts, inlier_im1, K_new)

            if T is not None:
                T_local = T_local_anchor @ T_base_cam @ np.linalg.inv(T)
                pose = se3_to_pose(T_local)
                if method == 'mast3r':
                    median_depth = np.median(mast3r_pts[:, 2]) if mast3r_pts.size > 0 else np.nan
                    pos_error, x_error, y_error, z_error, rot_error, _ = compute_statistics(se3_to_pose(T), pose_gt)
                    statistics = [n_matches, n_inliers, n_overlapping, median_depth, x_error, y_error, z_error, pos_error, rot_error, distance_anchor_query]
                elif method == 'lidar':
                    median_depth = np.median(lidar_pts[:, 2]) if lidar_pts.size > 0 else np.nan
                    pos_error, x_error, y_error, z_error, rot_error, _ = compute_statistics(se3_to_pose(T), pose_gt)
                    statistics = [n_matches, n_inliers, n_overlapping,  median_depth, x_error, y_error, z_error, pos_error, rot_error, distance_anchor_query]
                else:
                    
                    median_depth = np.median(scaled_pts[:, 2]) if scaled_pts.size > 0 else np.nan
                    pos_error, x_error, y_error, z_error, rot_error, pointmap_error = compute_statistics(
                        se3_to_pose(T), pose_gt, scaled_pts, lidar_pts)
                    statistics = [n_matches, n_inliers, n_overlapping,  median_depth, x_error, y_error, z_error, pos_error, rot_error, distance_anchor_query, pointmap_error, *np.ravel(scale)] 
                # print(f"Method: {method}, Statistics: {len(statistics), statistics}")
                save_result(output_file, anchor_seq, query_seq, query_idx, anchor_idx, pose, statistics, status="OK")
            else:
                median_depth =  np.nan
                save_failed_result(output_file, anchor_seq, query_seq, query_idx, anchor_idx, n_matches, n_inliers, median_depth, stat_len)
    except Exception as e:
        print(f"Exception for pair {anchor_idx}, {query_idx}: {e}")
        for method, stat_len, output_file in method_specs:
            save_failed_result(output_file, anchor_seq, query_seq, query_idx, anchor_idx, np.nan, np.nan, np.nan, stat_len)
def main():
    args = parse_args()
    dataset = vbrDataset(args.dataset, args.gt)
    calib = load_calibration(args.calib)
    K = calib['cam_l']['K']
    T_base_cam = calib['cam_l']['T_base_cam']
    T_cam_lidar = calib['cam_l']['T_cam_lidar']

    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)

    pairs_df = pd.read_csv(args.pairs_csv)
    processed_pairs = load_processed_pairs(args.temp_file)

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Estimating poses"):
        anchor_idx, query_idx = int(row['anchor_idx']), int(row['query_idx'])
        anchor_seq, query_seq = int(row['anchor_seq']), int(row['query_seq'])
        if (anchor_idx, query_idx) in processed_pairs:
            continue  # skip already processed
        anchor = dataset[anchor_idx]
        query = dataset[query_idx]
        process_pair(model, anchor, query, anchor_seq, query_seq, anchor_idx, query_idx, K, T_base_cam, T_cam_lidar, args)
        mark_pair_processed(args.temp_file, anchor_idx, query_idx)

if __name__ == "__main__":
    main()
