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

from my_vbr_utils.vbr_dataset import vbrInterpolatedDataset, load_calibration, get_paths_from_scene
from my_utils.my_vbr_dataset import generate_depth_and_scene_maps
from my_utils.mast3r_utils import (
    get_master_output, get_mast3r_image_shape, scale_intrinsics, overlap,
    solve_pnp, quaternion_rotational_error, recompute_intrinsics
)
from my_utils.scaling import scale_pnp, compute_scaled_points
from my_utils.transformations import pose_to_se3, se3_to_pose

METHOD_CONFIGS = {
    'mast3r': {'has_pointmap_error': True, 'has_scale': False, 'filename_suffix': 'mast3r'},
    'lidar': {'has_pointmap_error': False, 'has_scale': False, 'filename_suffix': 'lidar'},
    'v4': {'has_pointmap_error': True, 'has_scale': True, 'scale_type': 'vector', 'filename_suffix': 'mast3r_scaled_v4'},
    'v3': {'has_pointmap_error': True, 'has_scale': True, 'scale_type': 'scalar', 'filename_suffix': 'mast3r_scaled_v3'},
    'v2': {'has_pointmap_error': True, 'has_scale': True, 'scale_type': 'vector', 'filename_suffix': 'mast3r_scaled_v2'},
    'icp': {'has_pointmap_error': True, 'has_scale': True, 'scale_type': 'scalar', 'filename_suffix': 'mast3r_scaled_icp'},
}

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate poses from anchor-query pairs CSV.")
    parser.add_argument('--dataset_scenes', nargs='+', type=str, required=True, help='List of dataset scenes')
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to base folder where vbr dataset is saved')
    parser.add_argument('--pairs_path', type=str, required=True, help='Path to directory containing pairs files.  Will look for <scene>_pairs.txt')
    parser.add_argument('--split', type=str, required=True, help='Split to use from pairs: all, test, train, val')
    parser.add_argument('--conf_percentile', type=float, default=0.0,help='Drop bottom n%% of confidence values. Default=0 (disabled).')
    parser.add_argument('--output_root', type=str, required=True, help='Root directory for output files. Subdirectories will be created for each scene.')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, required=True, help="Path to Model Checkpoint/Pre-trained Weights")
    parser.add_argument('--min_inliers', type=int, default=200, help='Minimum inliers required for a match to be valid')
    parser.add_argument('--temp_file_prefix', type=str, required=True, help='Prefix for temp file.  Will create <prefix>_<scene>_processed_pairs.txt')
    return parser.parse_args()

def ensure_dir_exists(filepath):
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

def create_csv_with_headers(output_file, method_config):
    if not os.path.exists(output_file):
        ensure_dir_exists(output_file)
        
        headers = [
            'query_idx', 'anchor_idx', 'status', 'n_matches', 'n_inliers', 
            'n_overlapping', 'median_depth', 'x_error', 'y_error', 'z_error', 
            'pos_error', 'rot_error', 'distance_anchor_query'
        ]
        
        if method_config['has_pointmap_error']:
            headers.extend(['pointmap_error', 'pointmap_error_x', 'pointmap_error_y', 'pointmap_error_z'])
        
        if method_config['has_scale']:
            if method_config.get('scale_type') == 'vector':
                headers.extend(['scale_x', 'scale_y', 'scale_z'])
            else:
                headers.append('scale')
        
        headers.extend(['pose_x', 'pose_y', 'pose_z', 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw'])
        
        pd.DataFrame(columns=headers).to_csv(output_file, index=False)

def save_result_csv(output_file, query_idx, anchor_idx, pose, statistics, method, status="OK"):
    config = METHOD_CONFIGS[method]
    
    row = [query_idx, anchor_idx, status] + list(statistics[:10])
    
    if config['has_pointmap_error']:
        row.extend(statistics[10:14] if len(statistics) > 13 else [np.nan]*4)
    
    if config['has_scale']:
        start_idx = 14 if config['has_pointmap_error'] else 10
        if config.get('scale_type') == 'vector':
            row.extend(statistics[start_idx:start_idx+3] if len(statistics) > start_idx+2 else [np.nan]*3)
        else:
            row.append(statistics[start_idx] if len(statistics) > start_idx else np.nan)
    
    row.extend(pose)
    
    pd.DataFrame([row]).to_csv(output_file, mode='a', header=False, index=False)

def save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches=np.nan, n_inliers=np.nan, n_overlapping=np.nan):
    pose = [np.nan] * 7
    statistics = [n_matches, n_inliers, n_overlapping] + [np.nan] * 17
    save_result_csv(output_file, query_idx, anchor_idx, pose, statistics, method, status="FAILED")

def load_processed_pairs(filename):
    if not os.path.exists(filename):
        return set()
    with open(filename, "r") as f:
        return set(tuple(map(int, line.strip().split(','))) for line in f)

def mark_pair_processed(filename, anchor_idx, query_idx):
    ensure_dir_exists(filename)
    with open(filename, "a") as f:
        f.write(f"{anchor_idx},{query_idx}\n")

def compute_statistics(pose, gt_pose):
    pos_error = np.linalg.norm(np.array(pose[:3]) - np.array(gt_pose[:3]))
    x_error, y_error, z_error = np.abs(np.array(pose[:3]) - np.array(gt_pose[:3]))
    rot_error = quaternion_rotational_error(np.array(pose[3:]), np.array(gt_pose[3:]))
    return pos_error, x_error, y_error, z_error, rot_error

def compute_pointmap_error(pts3d_im0, scene_map):
    """Compute pointmap error using your method."""
    valid_mask = np.isfinite(scene_map).all(axis=2) & np.isfinite(pts3d_im0).all(axis=2)
    pts3d_error = np.abs(pts3d_im0[valid_mask] - scene_map[valid_mask])
    if len(pts3d_error) == 0:
        return [np.nan] * 4
    
    median_error = np.median(np.linalg.norm(pts3d_error, axis=1))
    median_error_x, median_error_y, median_error_z = np.median(pts3d_error, axis=0)
    return [median_error, median_error_x, median_error_y, median_error_z]

def compute_pointmap_abs_rel_error(pts3d_im0, scene_map):
    """Compute Absolute Relative Error (AbsRel) overall and per axis."""
    valid_mask = np.isfinite(scene_map).all(axis=2) & np.isfinite(pts3d_im0).all(axis=2)

    pred_pts = pts3d_im0[valid_mask]
    gt_pts = scene_map[valid_mask]

    if len(pred_pts) == 0:
        return [np.nan] * 4

    abs_error = np.abs(pred_pts - gt_pts)  # shape (N, 3)
    gt_abs = np.abs(gt_pts)

    # Mask to avoid division by zero on all axes
    nonzero_mask = gt_abs > 1e-6

    # Per-axis relative error, safely computed only where GT > 1e-6
    rel_error_per_axis = np.zeros_like(abs_error)
    rel_error_per_axis[nonzero_mask] = abs_error[nonzero_mask] / gt_abs[nonzero_mask]

    # Mean per-axis absolute relative errors
    abs_rel_x = np.mean(rel_error_per_axis[:, 0])
    abs_rel_y = np.mean(rel_error_per_axis[:, 1])
    abs_rel_z = np.mean(rel_error_per_axis[:, 2])

    # For overall error compute Euclidean norms once
    diff_norm = np.linalg.norm(pred_pts - gt_pts, axis=1)
    gt_norm = np.linalg.norm(gt_pts, axis=1)

    # Mask for valid gt_norm to avoid division by zero
    valid_gt_norm = gt_norm > 1e-6

    abs_rel_overall = np.mean(diff_norm[valid_gt_norm] / gt_norm[valid_gt_norm])

    return [abs_rel_overall, abs_rel_x, abs_rel_y, abs_rel_z]

def prepare_geometric_data(anchor, K, T_cam_lidar):
    img = cv2.imread(anchor['image'])
    H, W = img.shape[:2]
    mast3r_w, mast3r_h = get_mast3r_image_shape(W, H)
    K_new = recompute_intrinsics(K, (W, H), size=512)
    depth_map, scene_map = generate_depth_and_scene_maps(
        anchor['lidar_points'], K_new, T_cam_lidar, (mast3r_h, mast3r_w)
    )
    return K_new, depth_map, scene_map


def apply_conf_threshold(pts3d_im0, inlier_im0, inlier_im1, conf_im0, percentile):
    """Apply the same confidence threshold to both dense pts3d and sparse inliers."""
    thr = np.percentile(conf_im0[~np.isnan(conf_im0)], percentile)

    # 1) Dense map masking
    mask_dense = conf_im0 >= thr
    pts3d_im0_filtered = np.where(mask_dense[..., None], pts3d_im0, np.nan)

    # 2) Inlier filtering
    if inlier_im0 is not None and len(inlier_im0) > 0:
        u = np.clip(inlier_im0[:, 0].astype(int), 0, conf_im0.shape[1]-1)
        v = np.clip(inlier_im0[:, 1].astype(int), 0, conf_im0.shape[0]-1)
        keep = conf_im0[v, u] >= thr
        inlier_im0 = inlier_im0[keep]
        inlier_im1 = inlier_im1[keep]

    return pts3d_im0_filtered, inlier_im0, inlier_im1


def estimate_pose_for_method(method, inlier_im0, inlier_im1, pts3d_im0, scene_map, depth_map, K_new):
    if method == 'lidar':
        # Only method that requires overlap
        valid_mast3r_uv, valid_lidar_uv, matched_idx = overlap(inlier_im0, depth_map, max_pixel_dist=2)
        if len(matched_idx)==0:
            return None, None, None
        lidar_pts = scene_map[valid_lidar_uv[:, 1], valid_lidar_uv[:, 0]]
        T = solve_pnp(lidar_pts, inlier_im1[matched_idx], K_new)
        return T, lidar_pts, None

    elif method == 'mast3r':
        mast3r_pts = pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]]
        T = solve_pnp(mast3r_pts, inlier_im1, K_new)
        return T, mast3r_pts, None

    else:  # scaled methods
        valid_mask = np.isfinite(scene_map).all(axis=2) & np.isfinite(pts3d_im0).all(axis=2)
        mast3r_all, lidar_all = pts3d_im0[valid_mask], scene_map[valid_mask]
        scaled_pts, scale = compute_scaled_points(method, mast3r_all, lidar_all)
        if scaled_pts is None or scale is None:
            return None, None, None

        mast3r_inliers = pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]] * scale
        T = solve_pnp(mast3r_inliers, inlier_im1, K_new)
        return T, mast3r_inliers, scale


def process_pair(model, anchor, query, anchor_idx, query_idx, K, T_base_cam, T_cam_lidar, args, output_files):
    try:
        output = get_master_output(model, args.device, anchor['image'], query['image'], visualize=False, verbose=False)
        matches_im0, matches_im1, pts3d_im0 = output[:3]
        conf_im0 = output[4] if args.conf_percentile > 0 else None
        if len(matches_im0) < 8:
            for method, output_file in output_files.items():
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, len(matches_im0), 0)
            return

        F, mask_f = cv2.findFundamentalMat(matches_im0, matches_im1, cv2.FM_RANSAC, 1, 0.99)
        if mask_f is None:
            for method, output_file in output_files.items():
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, len(matches_im0), 0)
            return

        inlier_mask = mask_f.ravel().astype(bool)
        inlier_im0, inlier_im1 = matches_im0[inlier_mask], matches_im1[inlier_mask]

        n_inliers = len(inlier_im0)

        if conf_im0 is not None:
            pts3d_im0, inlier_im0, inlier_im1 = apply_conf_threshold(
                pts3d_im0, inlier_im0, inlier_im1, conf_im0, args.conf_percentile
            )

        n_matches = len(inlier_im0)

        if n_inliers < args.min_inliers:
            for method, output_file in output_files.items():
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches, n_inliers)
            return

        # Prepare depth/scene maps
        K_new, depth_map, scene_map = prepare_geometric_data(anchor, K, T_cam_lidar)

        # Compute GT
        T_anchor = pose_to_se3(anchor['pose'])
        T_query = pose_to_se3(query['pose'])
        T_gt = np.linalg.inv(T_query @ T_base_cam) @ T_anchor @ T_base_cam
        pose_gt = se3_to_pose(T_gt)
        distance_anchor_query = np.linalg.norm(np.array(anchor['pose'][:3]) - np.array(query['pose'][:3]))

        # Process each method
        for method, output_file in output_files.items():
            T, pts_for_depth, scale = estimate_pose_for_method(
                method, inlier_im0, inlier_im1, pts3d_im0, scene_map, depth_map, K_new
            )

            if T is None:
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches, n_inliers)
                continue

            T_local = T_anchor @ T_base_cam @ np.linalg.inv(T)
            pose = se3_to_pose(T_local)

            median_depth = np.median(pts_for_depth[:, 2]) if pts_for_depth is not None else np.nan
            pos_error, x_error, y_error, z_error, rot_error = compute_statistics(se3_to_pose(T), pose_gt)

            # Base stats
            statistics = [
                n_matches, n_inliers,
                (len(pts_for_depth) if method == 'lidar' else np.nan),  # n_overlapping only for lidar
                median_depth, x_error, y_error, z_error, pos_error, rot_error, distance_anchor_query
            ]

            config = METHOD_CONFIGS[method]
            if config['has_pointmap_error']:
                scaled_pts3d = pts3d_im0 * scale if scale is not None else pts3d_im0
                pointmap_errors = compute_pointmap_abs_rel_error(scaled_pts3d, scene_map)
                statistics.extend(pointmap_errors)

            if config['has_scale'] and scale is not None:
                statistics.extend(np.ravel(scale))

            save_result_csv(output_file, query_idx, anchor_idx, pose, statistics, method)

    except Exception as e:
        print(f"Exception for pair {anchor_idx}, {query_idx}: {e}")
        for method, output_file in output_files.items():
            print(method)
            save_failed_result_csv(output_file, query_idx, anchor_idx, method)


def main():
    args = parse_args()
    # Load model
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(args.model_path).to(args.device)
    
    # Process each scene
    for dataset_scene in args.dataset_scenes:
        print(f"Processing scene: {dataset_scene}")
        
        # Load dataset and calibration
        dataset = vbrInterpolatedDataset(dataset_root_dir=args.dataset_root, scene_name=dataset_scene)
        calib_path = get_paths_from_scene(args.dataset_root, dataset_scene)[-1]
        calib = load_calibration(calib_path)
        K, T_base_cam, T_cam_lidar = calib['cam_l']['K'], calib['cam_l']['T_base_cam'], calib['cam_l']['T_cam_lidar']
        
        # Load pairs
        pairs_file = os.path.join(args.pairs_path, f"{dataset_scene}", f"{args.split}_pairs.txt")
        pairs = []
        with open(pairs_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((int(parts[0]), int(parts[1])))
        
        temp_file = f"{args.temp_file_prefix}_{dataset_scene}_processed_pairs.txt"
        processed_pairs = load_processed_pairs(temp_file)
        
        # Create output files
        output_files = {}
        for method, config in METHOD_CONFIGS.items():
            output_dir = os.path.join(args.output_root, dataset_scene)
            output_file = os.path.join(output_dir, f"{config['filename_suffix']}.csv")
            output_files[method] = output_file
            create_csv_with_headers(output_file, config)
        
        # Process pairs
        for anchor_idx, query_idx in tqdm(pairs, desc=f"Estimating poses for {dataset_scene}"):
            if (anchor_idx, query_idx) in processed_pairs:
                continue
            
            anchor, query = dataset[anchor_idx], dataset[query_idx]
            process_pair(model, anchor, query, anchor_idx, query_idx, K, T_base_cam, T_cam_lidar, args, output_files)
            mark_pair_processed(temp_file, anchor_idx, query_idx)

if __name__ == "__main__":
    main()