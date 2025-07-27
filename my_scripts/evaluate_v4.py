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
    solve_pnp, quaternion_rotational_error
)
from my_utils.scaling import scale_pnp, compute_scaled_points
from my_utils.transformations import pose_to_se3, se3_to_pose

METHOD_CONFIGS = {
    'mast3r': {
        'has_pointmap_error': True,
        'has_scale': False,
        'filename_suffix': 'mast3r'
    },
    'lidar': {
        'has_pointmap_error': False,
        'has_scale': False,
        'filename_suffix': 'lidar'
    },
    'inliers_mast3r': {  
        'has_pointmap_error': False,
        'has_scale': False,
        'filename_suffix': 'inliers_mast3r'
    },
    'v4': {
        'has_pointmap_error': True,
        'has_scale': True,
        'scale_type': 'vector',  
        'filename_suffix': 'mast3r_scaled_v4'
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Estimate poses from anchor-query pairs CSV.")
    parser.add_argument('--dataset_scene', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, required=True, help='Path to base folder where vbr dataset is saved')
    parser.add_argument('--pairs_path', type=str, required=True, help='anchor_idx,query_idx')
    parser.add_argument('--output_prefix', type=str, default='estimates')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, required=True, help="Path to Model Checkpoint/Pre-trained Weights")
    parser.add_argument('--min_inliers', type=int, default=200, help='Minimum inliers required for a match to be valid')
    parser.add_argument('--temp_file', type=str, required=True, help='File to log pairs that have already been processed')
    return parser.parse_args()

def ensure_dir_exists(filepath):
    """Create directory for file if it doesn't exist."""
    dirpath = os.path.dirname(filepath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

def create_csv_with_headers(output_file, method_config):
    """Create CSV file with appropriate headers based on method configuration."""
    if not os.path.exists(output_file):
        ensure_dir_exists(output_file)
        
        base_headers = [
            'query_idx', 'anchor_idx', 'status', 'n_matches', 'n_inliers', 
            'n_overlapping', 'median_depth', 'x_error', 'y_error', 'z_error', 
            'pos_error', 'rot_error', 'distance_anchor_query'
        ]
        
        if method_config['has_pointmap_error']:
            base_headers.extend(['pointmap_error', 'pointmap_error_x', 'pointmap_error_y', 'pointmap_error_z'])
        
        if method_config['has_scale']:
            if method_config['scale_type'] == 'vector':
                base_headers.extend(['scale_x', 'scale_y', 'scale_z'])
            elif method_config['scale_type'] == 'scalar':
                base_headers.append('scale')
        
        # Add pose columns
        pose_headers = ['pose_x', 'pose_y', 'pose_z', 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw']
        headers = base_headers + pose_headers
        
        df = pd.DataFrame(columns=headers)
        df.to_csv(output_file, index=False)

def save_result_csv(output_file, query_idx, anchor_idx, pose, statistics, method, status="OK"):
    """Save result to CSV file."""
    # Base row data
    row_data = {
        'query_idx': query_idx,
        'anchor_idx': anchor_idx,
        'status': status,
        'n_matches': statistics[0] if len(statistics) > 0 else np.nan,
        'n_inliers': statistics[1] if len(statistics) > 1 else np.nan,
        'n_overlapping': statistics[2] if len(statistics) > 2 else np.nan,
        'median_depth': statistics[3] if len(statistics) > 3 else np.nan,
        'x_error': statistics[4] if len(statistics) > 4 else np.nan,
        'y_error': statistics[5] if len(statistics) > 5 else np.nan,
        'z_error': statistics[6] if len(statistics) > 6 else np.nan,
        'pos_error': statistics[7] if len(statistics) > 7 else np.nan,
        'rot_error': statistics[8] if len(statistics) > 8 else np.nan,
        'distance_anchor_query': statistics[9] if len(statistics) > 9 else np.nan,
    }

    config = METHOD_CONFIGS[method]

    if config['has_pointmap_error']:
        row_data['pointmap_error'] = statistics[10] if len(statistics) > 10 else np.nan
        row_data['pointmap_error_x'] = statistics[11] if len(statistics) > 11 else np.nan
        row_data['pointmap_error_y'] = statistics[12] if len(statistics) > 12 else np.nan
        row_data['pointmap_error_z'] = statistics[13] if len(statistics) > 13 else np.nan

    if config['has_scale']:
        if config['scale_type'] == 'vector':
            row_data['scale_x'] = statistics[14] if len(statistics) > 14 else np.nan
            row_data['scale_y'] = statistics[15] if len(statistics) > 15 else np.nan
            row_data['scale_z'] = statistics[16] if len(statistics) > 16 else np.nan
        elif config['scale_type'] == 'scalar':
            row_data['scale'] = statistics[14] if len(statistics) > 14 else np.nan

    # Add pose data
    pose_names = ['pose_x', 'pose_y', 'pose_z', 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw']
    for i, name in enumerate(pose_names):
        row_data[name] = pose[i] if i < len(pose) else np.nan

    df = pd.DataFrame([row_data])
    df.to_csv(output_file, mode='a', header=False, index=False)



def save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches=np.nan, n_inliers=np.nan, n_overlapping=np.nan):
    """Save failed result to CSV file."""
    pose = [np.nan] * 7
    statistics = [n_matches, n_inliers, n_overlapping] + [np.nan] * 7  # Fill remaining with NaN
    save_result_csv(output_file, query_idx, anchor_idx, pose, statistics, method, status="FAILED")

def save_exception_result_csv(output_file, query_idx, anchor_idx, method):
    """Save exception result to CSV file."""
    pose = [np.nan] * 7
    statistics = [np.nan] * 10  # All stats as NaN
    save_result_csv(output_file, query_idx, anchor_idx, pose, statistics, method, status="EXCEPTION")


def load_processed_pairs(filename):
    """Load set of processed pairs from file."""
    if not os.path.exists(filename):
        return set()
    with open(filename, "r") as f:
        return set(tuple(map(int, line.strip().split(','))) for line in f)

def mark_pair_processed(filename, anchor_idx, query_idx):
    """Mark pair as processed by appending to file."""
    ensure_dir_exists(filename)
    with open(filename, "a") as f:
        f.write(f"{anchor_idx},{query_idx}\n")

def compute_statistics(pose, gt_pose, mast3r_pts=None, lidar_pts=None):
    """Compute pose error statistics."""
    pos_error = np.linalg.norm(np.array(pose[:3]) - np.array(gt_pose[:3]))
    x_error, y_error, z_error = np.abs(np.array(pose[:3]) - np.array(gt_pose[:3]))
    rot_error = quaternion_rotational_error(np.array(pose[3:]), np.array(gt_pose[3:]))
    
    pointmap_error = np.nan
    pointmap_error_x = np.nan
    pointmap_error_y = np.nan
    pointmap_error_z = np.nan

    if mast3r_pts is not None and lidar_pts is not None and len(mast3r_pts) == len(lidar_pts):
        pointmap_diff = mast3r_pts - lidar_pts
        pointmap_error = np.mean(np.linalg.norm(pointmap_diff, axis=1))
        pointmap_error_x, pointmap_error_y, pointmap_error_z = np.mean(np.abs(pointmap_diff), axis=0)
    
    return pos_error, x_error, y_error, z_error, rot_error, pointmap_error, pointmap_error_x, pointmap_error_y, pointmap_error_z
    

def get_matches(model, device, anchor_img, query_img):
    """Get feature matches between anchor and query images."""
    output = get_master_output(model, device, anchor_img, query_img, visualize=False, verbose=False)
    return output[0], output[1], output[2]

def filter_inliers(matches_im0, matches_im1):
    """Filter matches using fundamental matrix RANSAC."""
    if len(matches_im0) < 8:
        return np.empty((0, 2)), np.empty((0, 2))
    F, mask_f = cv2.findFundamentalMat(matches_im0, matches_im1, cv2.FM_RANSAC, 0.5, 0.99)
    if mask_f is None:
        return np.empty((0, 2)), np.empty((0, 2))
    inlier_mask = mask_f.ravel().astype(bool)
    return matches_im0[inlier_mask], matches_im1[inlier_mask]

def get_intrinsics_and_maps(anchor, K, T_cam_lidar):
    """Get scaled intrinsics and depth/scene maps for anchor image."""
    img = cv2.imread(anchor['image'])
    H, W = img.shape[:2]
    mast3r_w, mast3r_h = get_mast3r_image_shape(W, H)
    K_new = scale_intrinsics(K, W, H, mast3r_w, mast3r_h)
    depth_map, scene_map = generate_depth_and_scene_maps(
        anchor['lidar_points'], K_new, T_cam_lidar, (mast3r_h, mast3r_w)
    )
    return K_new, depth_map, scene_map

def get_overlap_points(inlier_im0, depth_map):
    """Find overlapping points between image features and LiDAR depth map."""
    valid_mast3r_uv, valid_lidar_uv, matched_idx = overlap(inlier_im0, depth_map, max_pixel_dist=2)
    if len(matched_idx) > 0:
        return matched_idx, valid_lidar_uv
    return np.empty((0, 2)), np.empty((0, 2))

def compute_ground_truth_pose(anchor, query, T_base_cam):
    """Compute ground truth relative pose between anchor and query."""
    T_local_anchor = pose_to_se3(anchor['pose'])
    T_local_query = pose_to_se3(query['pose'])
    T_query_anchor_gt = np.linalg.inv(T_local_query @ T_base_cam) @ T_local_anchor @ T_base_cam
    return se3_to_pose(T_query_anchor_gt)

def compute_distance(pose1, pose2):
    """Compute Euclidean distance between two poses."""
    return np.linalg.norm(np.array(pose1[:3]) - np.array(pose2[:3]))

def estimate_and_evaluate_pose(method, mast3r_pts, lidar_pts, inlier_im1, K_new,
                              pose_gt, distance_anchor_query, n_matches, n_inliers, n_overlapping,
                              T_local_anchor, T_base_cam):
    """Estimate pose using specified method and compute statistics."""

    # Estimate pose based on method
    if method == 'mast3r':
        T = solve_pnp(mast3r_pts, inlier_im1, K_new)
        pts_for_depth = mast3r_pts
        scale = None
    elif method == 'lidar':
        T = solve_pnp(lidar_pts, inlier_im1, K_new)
        pts_for_depth = lidar_pts
        scale = None
    elif method == 'inliers_mast3r':  
        T = solve_pnp(mast3r_pts, inlier_im1, K_new)
        pts_for_depth = mast3r_pts
        scale = None
    else:
        pts_for_depth, scale, T = scale_pnp(method, mast3r_pts, lidar_pts, inlier_im1, K_new)

    if T is None:
        return {'success': False}

    # Compute pose in local coordinate system
    T_local = T_local_anchor @ T_base_cam @ np.linalg.inv(T)
    pose = se3_to_pose(T_local)

    pos_error, x_error, y_error, z_error, rot_error, pointmap_error, pointmap_error_x, pointmap_error_y, pointmap_error_z = compute_statistics(
        se3_to_pose(T), pose_gt,
        mast3r_pts if method != 'lidar' else pts_for_depth,
        lidar_pts
    )

    median_depth = np.median(pts_for_depth[:, 2]) if pts_for_depth is not None and pts_for_depth.size > 0 else np.nan

    # Build statistics list based on method configuration
    statistics = [
        n_matches, n_inliers, n_overlapping, median_depth,
        x_error, y_error, z_error, pos_error, rot_error, distance_anchor_query
    ]

    config = METHOD_CONFIGS[method]
    if config['has_pointmap_error']:
            statistics.extend([pointmap_error, pointmap_error_x, pointmap_error_y, pointmap_error_z])
    if config['has_scale'] and scale is not None:
        statistics.extend(np.ravel(scale))

    return {'success': True, 'pose': pose, 'statistics': statistics}

def process_pair(model, anchor, query, anchor_idx, query_idx, K, T_base_cam, T_cam_lidar, args, output_files):
    """Process a single anchor-query pair for all methods."""
    try:
        # 1. Feature matching
        matches_im0, matches_im1, pts3d_im0 = get_matches(model, args.device, anchor['image'], query['image'])
        inlier_im0_initial, inlier_im1_initial = filter_inliers(matches_im0, matches_im1)
        n_matches, n_inliers = len(matches_im0), len(inlier_im0_initial)

        # Early exit conditions
        if n_inliers <= args.min_inliers or inlier_im0_initial.size == 0:
            for method, output_file in output_files.items():
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches, n_inliers)
            return

        # 2. Prepare geometric data
        K_new, depth_map, scene_map = get_intrinsics_and_maps(anchor, K, T_cam_lidar)

        # 3. Compute ground truth and distance
        pose_gt = compute_ground_truth_pose(anchor, query, T_base_cam)
        distance_anchor_query = compute_distance(anchor['pose'], query['pose'])
        T_local_anchor = pose_to_se3(anchor['pose'])

        # 4. Process each method
        for method, output_file in output_files.items():
            inlier_im0 = inlier_im0_initial.copy()
            inlier_im1 = inlier_im1_initial.copy()
            if method == 'inliers_mast3r':
                # Use all inliers, skip overlap calculation
                mast3r_pts = pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]]
                lidar_pts = None  # Not used for this method
                n_overlapping = np.nan  # Not used for this method
            else:
                # Original processing for other methods
                matched_idx, valid_lidar_uv = get_overlap_points(inlier_im0, depth_map)
                n_overlapping = len(matched_idx)

                if matched_idx.size == 0:
                    save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches, n_inliers, n_overlapping)
                    continue
                
                inlier_im0 = inlier_im0[matched_idx]
                inlier_im1 = inlier_im1[matched_idx]
                mast3r_pts = pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]]
                lidar_pts = scene_map[valid_lidar_uv[:, 1], valid_lidar_uv[:, 0]]

                if len(inlier_im0) < 4:
                    save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches, n_inliers, n_overlapping)
                    continue 

            result = estimate_and_evaluate_pose(
                method, mast3r_pts, lidar_pts, inlier_im1, K_new,
                pose_gt, distance_anchor_query, n_matches, n_inliers, n_overlapping,
                T_local_anchor, T_base_cam
            )

            if result['success']:
                save_result_csv(output_file, query_idx, anchor_idx, result['pose'], result['statistics'], method)
            else:
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches, n_inliers, n_overlapping)

    except Exception as e:
        print(f"Exception for pair {anchor_idx}, {query_idx}: {e}")
        for method, output_file in output_files.items():
            save_exception_result_csv(output_file, query_idx, anchor_idx, method)
def main():
    args = parse_args()
    
    # Load dataset and calibration
    dataset = vbrInterpolatedDataset(dataset_root_dir=args.dataset_root, scene_name=args.dataset_scene)
    calib_path = get_paths_from_scene(args.dataset_root, args.dataset_scene)[-1]
    calib = load_calibration(calib_path)
    K = calib['cam_l']['K']
    T_base_cam = calib['cam_l']['T_base_cam']
    T_cam_lidar = calib['cam_l']['T_cam_lidar']

    # Load MASt3R model
    from mast3r.model import AsymmetricMASt3R
    model_path = args.model_path
    model = AsymmetricMASt3R.from_pretrained(model_path).to(args.device)

    # Read anchor and query indices from the txt file
    pairs = []
    with open(args.pairs_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                anchor_idx, query_idx = int(parts[0]), int(parts[1])
                pairs.append((anchor_idx, query_idx))
    # Load pairs and processed pairs
    processed_pairs = load_processed_pairs(args.temp_file)

    # Create output files for each method
    output_files = {}
    for method, config in METHOD_CONFIGS.items():
        output_file = f"{args.output_prefix}_{config['filename_suffix']}.csv"
        output_files[method] = output_file
        create_csv_with_headers(output_file, config)


    for anchor_idx, query_idx in tqdm(pairs, total=len(pairs), desc="Estimating poses"):
        if (anchor_idx, query_idx) in processed_pairs:
            continue  # skip already processed
        anchor = dataset[anchor_idx]
        query = dataset[query_idx]
        process_pair(model, anchor, query, anchor_idx, query_idx, K, T_base_cam, T_cam_lidar, args, output_files)
        mark_pair_processed(args.temp_file, anchor_idx, query_idx)

if __name__ == "__main__":
    main()