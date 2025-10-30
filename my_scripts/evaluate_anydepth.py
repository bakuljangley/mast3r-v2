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
    get_master_output, get_mast3r_image_shape,
    solve_pnp, quaternion_rotational_error, recompute_intrinsics, apply_conf_threshold
)
from my_utils.transformations import pose_to_se3, se3_to_pose

METHOD_CONFIGS = {
    # 'anydepth80': {'has_pointmap_error': True, 'has_scale': False, 'depth_root': '/home/bjangley/VPR/depthanything_80'},
    # 'zoedepth': {'has_pointmap_error': True, 'has_scale': False, 'depth_root': '/home/bjangley/VPR/zoedepth_vbr'},
    'depthpro': {'has_pointmap_error': True, 'has_scale': False, 'depth_root': '/home/bjangley/VPR/depthpro_vbr'},
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
        if method_config.get('has_scale', False):
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
    if config.get('has_scale', False):
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
    """Median absolute 3D error over valid (finite) pixels."""
    if pts3d_im0 is None or scene_map is None:
        return [np.nan] * 4
    valid_mask = np.isfinite(scene_map).all(axis=2) & np.isfinite(pts3d_im0).all(axis=2)
    if not np.any(valid_mask):
        return [np.nan] * 4
    pts3d_error = np.abs(pts3d_im0[valid_mask] - scene_map[valid_mask])
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


def generate_scene_map_from_depth(depth_map, K, T_cam_lidar):
    """Backproject depth to 3D in the camera frame (no extra transform)."""
    if depth_map is None:
        return None
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map.astype(np.float32).reshape(-1)
    x = (u.reshape(-1) - K[0, 2]) * z / K[0, 0]
    y = (v.reshape(-1) - K[1, 2]) * z / K[1, 1]
    pts3d_cam = np.stack([x, y, z], axis=-1).reshape(H, W, 3)
    return pts3d_cam

def estimate_pose_for_method(inlier_im0, inlier_im1, scene_map, K_new):
    """
    Pose from cached scene_map (HxWx3). 
    """
    if scene_map is None or len(inlier_im0) < 4:
        return None, None, None
    u = inlier_im0[:, 0].astype(int)
    v = inlier_im0[:, 1].astype(int)
    pts3d = scene_map[v, u]  # direct indexing as requested
    valid = np.isfinite(pts3d).all(axis=1)  # not a bounds check
    if valid.sum() < 4:
        return None, None, None
    T = solve_pnp(pts3d[valid], inlier_im1[valid], K_new)
    return T, pts3d[valid], None

def process_pair(model, anchor, query, anchor_idx, query_idx, K, T_base_cam, T_cam_lidar, args, output_files, dataset_scene, scene_maps):
    try:
        # matches via MASt3R
        output = get_master_output(model, args.device, anchor['image'], query['image'], visualize=False, verbose=False)
        matches_im0, matches_im1, pts3d_im0 = output[0], output[1], output[2]
        conf_im0 = output[4] if args.conf_percentile > 0 else None
        if len(matches_im0) < 8:
            for method, output_file in output_files.items():
                print("Fail")
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, len(matches_im0), 0, np.nan)
            return

        F, mask_f = cv2.findFundamentalMat(matches_im0, matches_im1, cv2.FM_RANSAC, 1, 0.99)
        if mask_f is None:
            for method, output_file in output_files.items():
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, len(matches_im0), 0, np.nan)
            return
        

        inlier_mask = mask_f.ravel().astype(bool)
        inlier_im0, inlier_im1 = matches_im0[inlier_mask], matches_im1[inlier_mask]
        n_matches, n_inliers = len(matches_im0), len(inlier_im0)

        if conf_im0 is not None:
            pts3d_im0, inlier_im0, inlier_im1 = apply_conf_threshold(
                pts3d_im0, inlier_im0, inlier_im1, conf_im0, args.conf_percentile
            )

        n_overlapping = len(inlier_im0)
        if n_inliers < args.min_inliers:
            for method, output_file in output_files.items():
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches, n_inliers, n_overlapping)
            return

        # intrinsics resized to MASt3R size
        img = cv2.imread(anchor['image'])
        H, W = img.shape[:2]
        mast3r_w, mast3r_h = get_mast3r_image_shape(W, H)[:2]
        K_new = recompute_intrinsics(K, (W, H), size=512)

        # GT relative
        T_anchor = pose_to_se3(anchor['pose'])
        T_query  = pose_to_se3(query['pose'])
        T_gt = np.linalg.inv(T_query @ T_base_cam) @ T_anchor @ T_base_cam
        pose_gt = se3_to_pose(T_gt)
        distance_anchor_query = np.linalg.norm(np.array(anchor['pose'][:3]) - np.array(query['pose'][:3]))

        # LiDAR reference for error metrics
        _, lidar_scene_map = generate_depth_and_scene_maps(anchor['lidar_points'], K_new, T_cam_lidar, (mast3r_h, mast3r_w))

        for method, output_file in output_files.items():
            scene_map = scene_maps[method].get(anchor_idx)
            if scene_map is None:
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches, n_inliers, n_overlapping)
                continue

            T, pts_for_depth, _ = estimate_pose_for_method(inlier_im0, inlier_im1, scene_map, K_new)
            if T is None:
                save_failed_result_csv(output_file, query_idx, anchor_idx, method, n_matches, n_inliers, n_overlapping)
                continue

            # convert to local pose
            T_local = T_anchor @ T_base_cam @ np.linalg.inv(T)
            pose = se3_to_pose(T_local)

            median_depth = np.median(pts_for_depth[:, 2]) if pts_for_depth is not None and len(pts_for_depth) > 0 else np.nan
            pos_error, x_error, y_error, z_error, rot_error = compute_statistics(se3_to_pose(T), pose_gt)

            statistics = [
                n_matches, n_inliers,
                n_overlapping,  # n_overlapping logs number of matches lost because of conf_thresholding
                median_depth, x_error, y_error, z_error, pos_error, rot_error, distance_anchor_query
            ]

            config = METHOD_CONFIGS[method]
            if config['has_pointmap_error']:
                pointmap_errors = compute_pointmap_abs_rel_error(lidar_scene_map, scene_map)
                statistics.extend(pointmap_errors)

            save_result_csv(output_file, query_idx, anchor_idx, pose, statistics, method)

    except Exception as e:
        print(f"Exception for pair {anchor_idx}, {query_idx}: {e}")
        for method, output_file in output_files.items():
            save_failed_result_csv(output_file, query_idx, anchor_idx, method)

def main():
    args = parse_args()
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(args.model_path).to(args.device)

    for dataset_scene in args.dataset_scenes:
        print(f"Processing scene: {dataset_scene}")

        dataset = vbrInterpolatedDataset(dataset_root_dir=args.dataset_root, scene_name=dataset_scene)
        calib_path = get_paths_from_scene(args.dataset_root, dataset_scene)[-1]
        calib = load_calibration(calib_path)
        K, T_base_cam, T_cam_lidar = calib['cam_l']['K'], calib['cam_l']['T_base_cam'], calib['cam_l']['T_cam_lidar']

        pairs_file = os.path.join(args.pairs_path, f"{dataset_scene}", f"{args.split}_pairs.txt")
        print("Pairs Path: ", pairs_file)
        pairs = []
        with open(pairs_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    pairs.append((int(parts[0]), int(parts[1])))

        temp_file = f"{args.temp_file_prefix}_{dataset_scene}_processed_pairs.txt"
        processed_pairs = load_processed_pairs(temp_file)
        print("Number of pairs: ", len(pairs))

        output_files = {}
        for method in METHOD_CONFIGS:
            config = METHOD_CONFIGS[method]
            out_dir = os.path.join(args.output_root, dataset_scene)
            outfile = os.path.join(out_dir, f"{method}.csv")
            output_files[method] = outfile
            create_csv_with_headers(outfile, config)

        anchor_indices = set(pair[0] for pair in pairs)
        scene_maps = {method: {} for method in METHOD_CONFIGS}
        for method, config in METHOD_CONFIGS.items():
            depth_root = config['depth_root']
            for idx in anchor_indices:
                anchor = dataset[idx]
                image_idx = os.path.splitext(os.path.basename(anchor['image']))[0]
                depth_path = os.path.join(depth_root, dataset_scene, f"{image_idx}.npy")
                if os.path.exists(depth_path):
                    depth_map = np.load(depth_path, mmap_mode='r').astype(np.float32)
                    img = cv2.imread(anchor['image'])
                    H, W = img.shape[:2]
                    mast3r_w, mast3r_h = get_mast3r_image_shape(W, H)[:2]
                    resized = cv2.resize(depth_map, (mast3r_w, mast3r_h), interpolation=cv2.INTER_NEAREST)
                    K_new = recompute_intrinsics(K, (W, H), size=512)
                    scene_map = generate_scene_map_from_depth(resized, K_new, T_cam_lidar)
                    scene_maps[method][idx] = scene_map
                else:
                    scene_maps[method][idx] = None

        for anchor_idx, query_idx in tqdm(pairs, desc=f"Estimating poses for {dataset_scene}"):
            if (anchor_idx, query_idx) in processed_pairs:
                continue
            anchor, query = dataset[anchor_idx], dataset[query_idx]
            process_pair(model, anchor, query, anchor_idx, query_idx, K, T_base_cam, T_cam_lidar, args, output_files, dataset_scene, scene_maps)
            mark_pair_processed(temp_file, anchor_idx, query_idx)

        del scene_maps
        import gc
        gc.collect()
        print(f"Cleared depth maps for scene {dataset_scene}")

if __name__ == "__main__":
    main()
