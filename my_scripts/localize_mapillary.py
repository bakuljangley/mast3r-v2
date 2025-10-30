#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os
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
    get_master_output, get_mast3r_image_shape, overlap,
    quaternion_rotational_error, recompute_intrinsics, apply_conf_threshold
)
from my_utils.transformations import pose_to_se3, se3_to_pose, pnp_to_se3
from my_utils.mast3r_utils import getUTMzone, getRotationFromCompass, transformGlobalCoordinates

# === CONFIG ===
METHOD_CONFIGS = {
    'anydepth80': {
        'depth_root': '/home/bjangley/VPR/depthanything_mapillary',
        'filename_suffix': 'anydepth80',
        'has_pointmap_error': True,
        'has_scale': False
    },
    'lidar': {
        'filename_suffix': 'lidar',
        'has_pointmap_error': False,
        'has_scale': False
    }
}

# default folder to vbr_utils folder containing the gps aligned trajectories
VBR_UTILS = "/home/bjangley/VPR/mast3r-v2/my_vbr_utils/global_trajectory/"

# ----------------- args -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Mapillary pose estimation using MASt3R + predicted depths (sequence-wise)")
    p.add_argument('--dataset_scenes', nargs='+', required=True)
    p.add_argument('--dataset_root', required=True)
    p.add_argument('--mapillary_root', required=True)
    p.add_argument('--pairs_root', required=True)
    p.add_argument('--output_root', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--model_path', required=True)
    p.add_argument('--conf_percentile', type=float, default=0.0)
    p.add_argument('--min_inliers', type=int, default=200)
    p.add_argument('--temp_file_prefix', required=True)
    return p.parse_args()

# ----------------- io helpers -----------------
def ensure_dir_exists(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def create_csv_with_headers(filename):
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            # add mapillary_sequence into the schema
            f.write('mapillary_sequence,query_idx,anchor_idx,method,num_matches,num_inliers,lat,lon,compass_deg,anchor_lat,anchor_lon,status\n')

def save_failed_result_csv(filename, sequence, query_idx, anchor_idx, method, num_matches, num_inliers, compass_deg, anchor_lat=np.nan, anchor_lon=np.nan):
    with open(filename, 'a') as f:
        f.write(f"{sequence},{query_idx},{anchor_idx},{method},{num_matches},{num_inliers},nan,nan,{compass_deg},{anchor_lat},{anchor_lon},fail\n")

def save_successful_result_csv(filename, sequence, query_idx, anchor_idx, method, num_matches, num_inliers, lat, lon, compass_deg, anchor_lat, anchor_lon):
    with open(filename, 'a') as f:
        f.write(f"{sequence},{query_idx},{anchor_idx},{method},{num_matches},{num_inliers},{lat},{lon},{compass_deg},{anchor_lat},{anchor_lon},success\n")

def processed_pairs_path(temp_prefix, scene):
    # per-scene processed ledger, contains CSV lines: sequence,anchor_idx,query_id
    return f"{temp_prefix}_{scene}_processed_pairs.txt"

def load_processed_pairs(filename):
    if not os.path.exists(filename):
        return set()
    out = set()
    with open(filename, "r") as f:
        for line in f:
            seq, a, q = line.strip().split(',')
            out.add((seq, int(a), q))
    return out

def mark_pair_processed(filename, sequence, anchor_idx, query_id):
    ensure_dir_exists(filename)
    with open(filename, "a") as f:
        f.write(f"{sequence},{anchor_idx},{query_id}\n")

# ----------------- math/utils -----------------
def generate_scene_map_from_depth(depth_map, K):
    if depth_map is None:
        return None
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth_map.astype(np.float32).reshape(-1)
    x = (u.reshape(-1) - K[0, 2]) * z / K[0, 0]
    y = (v.reshape(-1) - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1).reshape(H, W, 3)

def solve_pnp(pts3d, pts2d, K, dist_coeffs=None):
    """
    Solve PnP given matched 3D-2D correspondences.
    Returns 4x4 SE(3) or None.
    """
    if pts3d is None or pts2d is None or len(pts3d) < 4 or len(pts2d) < 4:
        return None
    success, rvec, tvec, _ = cv2.solvePnPRansac(
        pts3d.astype(np.float32),
        pts2d.astype(np.float32),
        K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    return pnp_to_se3(rvec, tvec) if success else None

def estimate_pose_from_scene_map(inlier_im0, inlier_im1, scene_map, K_query_512, dist_coeffs):
    if scene_map is None or inlier_im0 is None or len(inlier_im0) < 4:
        return None, None
    u = inlier_im0[:, 0].astype(int)
    v = inlier_im0[:, 1].astype(int)
    pts3d = scene_map[v, u]
    T = solve_pnp(pts3d.astype(np.float32), inlier_im1.astype(np.float32), K_query_512, dist_coeffs)
    return T, pts3d

def camera_matrix_from_mapillary_f(f, W, H):
    return np.array([[f * W, 0, W / 2.0],
                     [0, f * H, H / 2.0],
                     [0, 0, 1.0]], dtype=np.float32)

def extract_gps(df, idx):
    def _find_col(df, candidates):
        cands = [c.lower() for c in candidates]
        for col in df.columns:
            low = col.lower()
            if any(key in low for key in cands):
                return col
        return None
    if len(df) == 0:
        raise ValueError("Empty trajectory file.")
    row = df.iloc[idx]
    lat_col = _find_col(df, ["lat"])
    lon_col = _find_col(df, ["lon", "long"])
    hdg_col = _find_col(df, ["heading", "yaw", "compass", "course"])
    if lat_col is None or lon_col is None:
        raise ValueError("Could not find latitude/longitude columns in trajectory CSV.")
    ref_lat = float(row[lat_col])
    ref_lon = float(row[lon_col])
    compass_deg = float(row[hdg_col]) if hdg_col is not None and pd.notnull(row[hdg_col]) else 0.0
    return ref_lat, ref_lon, compass_deg

# ----------------- core pair processing -----------------
def process_pair(model, device, anchor, query_img_path, query_idx,
                 K_cam, T_base_cam, T_cam_lidar, conf_percentile,
                 min_inliers, K_query_512, dist_coeffs, scene_map,
                 ref_lat, ref_lon, compass_deg, out_file, method, sequence):
    try:
        out = get_master_output(model, device, anchor['image'], query_img_path,
                                visualize=False, verbose=False)
        matches_im0, matches_im1, pts3d_im0 = out[0], out[1], out[2]
        conf_im0 = out[4] if conf_percentile > 0 else None

        if matches_im0 is None or len(matches_im0) < 8:
            save_failed_result_csv(out_file, sequence, query_idx, anchor['idx'], method, np.nan, np.nan, compass_deg, ref_lat, ref_lon)
            return

        F, mask = cv2.findFundamentalMat(matches_im0, matches_im1, cv2.FM_RANSAC, 1, 0.99)
        if mask is None:
            save_failed_result_csv(out_file, sequence, query_idx, anchor['idx'], method, len(matches_im0), np.nan, compass_deg, ref_lat, ref_lon)
            return

        inlier_mask = mask.ravel().astype(bool)
        inlier_im0, inlier_im1 = matches_im0[inlier_mask], matches_im1[inlier_mask]
        n_matches, n_inliers = len(matches_im0), len(inlier_im0)

        if conf_im0 is not None:
            pts3d_im0, inlier_im0, inlier_im1 = apply_conf_threshold(
                pts3d_im0, inlier_im0, inlier_im1, conf_im0, conf_percentile
            )

        img = cv2.imread(anchor['image'])
        H, W = img.shape[:2]
        mast3r_w, mast3r_h = get_mast3r_image_shape(W, H)[:2]
        K_anchor_512 = recompute_intrinsics(K_cam, (W, H), size=512)

        if method == 'lidar':
            depth_map, lidar_scene_map = generate_depth_and_scene_maps(
                anchor['lidar_points'], K_anchor_512, T_cam_lidar, (mast3r_h, mast3r_w)
            )
            valid_mast3r_uv, valid_lidar_uv, matched_idx = overlap(inlier_im0, depth_map, max_pixel_dist=2)
            if len(matched_idx) == 0:
                save_failed_result_csv(out_file, sequence, query_idx, anchor['idx'], method, n_matches, n_inliers, compass_deg, ref_lat, ref_lon)
                return
            lidar_pts = lidar_scene_map[valid_lidar_uv[:, 1], valid_lidar_uv[:, 0]]
            T = solve_pnp(lidar_pts, inlier_im1[matched_idx], K_query_512, dist_coeffs)
        else:
            if n_inliers < min_inliers or scene_map is None or K_query_512 is None:
                save_failed_result_csv(out_file, sequence, query_idx, anchor['idx'], method, n_matches, n_inliers, compass_deg, ref_lat, ref_lon)
                return
            T, _ = estimate_pose_from_scene_map(inlier_im0, inlier_im1, scene_map, K_query_512, dist_coeffs)

        if T is None:
            save_failed_result_csv(out_file, sequence, query_idx, anchor['idx'], method, n_matches, n_inliers, compass_deg, ref_lat, ref_lon)
            return

        R, t = T[:3, :3], T[:3, 3].reshape(3,)
        q_lat, q_lon, _ = transformGlobalCoordinates(R, t, ref_lat, ref_lon, compass_deg)

        save_successful_result_csv(out_file, sequence, query_idx, anchor['idx'], method, n_matches, n_inliers, q_lat, q_lon, compass_deg, ref_lat, ref_lon)

    except Exception:
        save_failed_result_csv(out_file, sequence, query_idx, anchor['idx'], method, np.nan, np.nan, compass_deg, ref_lat, ref_lon)

# ----------------- main -----------------
def main():
    args = parse_args()
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(args.model_path).to(args.device)

    for scene in args.dataset_scenes:
        print(f"\n[Scene] {scene}")

        # Load VBR scene and calibration
        dataset = vbrInterpolatedDataset(args.dataset_root, scene)
        calib_path = get_paths_from_scene(args.dataset_root, scene)[-1]
        calib = load_calibration(calib_path)
        K_cam = calib['cam_l']['K']
        T_base_cam = calib['cam_l']['T_base_cam']
        T_cam_lidar = calib['cam_l']['T_cam_lidar']

        # Load GPS trajectory (reference for global projection)
        gps_traj = os.path.join(VBR_UTILS, scene, 'trajectory.csv')
        traj_df = pd.read_csv(gps_traj)

        # Load pairs and group by sequence
        pairs_path = os.path.join(args.pairs_root, scene, 'results_top3_anchors_per_query.csv')
        if not os.path.exists(pairs_path):
            raise FileNotFoundError(f"[x] Missing pairs file: {pairs_path}")
        pairs_df = pd.read_csv(pairs_path)
        if 'sequence' not in pairs_df.columns:
            raise KeyError("pairs CSV must contain a 'sequence' column.")

        pairs_df['anchor_idx'] = pairs_df['anchor_idx'].astype(int)
        pairs_df['query_image'] = pairs_df['query_image'].astype(str)
        pairs_df['sequence'] = pairs_df['sequence'].astype(str)

        pairs_by_sequence = {seq: df for seq, df in pairs_df.groupby('sequence')}

        # Preload AnyDepth depth maps once per scene (for all anchors in pairs)
        anchor_indices = sorted(set(pairs_df['anchor_idx'].tolist()))
        scene_maps = {method: {} for method in METHOD_CONFIGS}
        print(f"[Preload] Depth maps for {len(anchor_indices)} anchors ...")
        for method, cfg in METHOD_CONFIGS.items():
            if method == 'lidar':
                continue
            for idx in anchor_indices:
                try:
                    anchor = dataset[idx]
                    img = cv2.imread(anchor['image'])
                    if img is None:
                        scene_maps[method][idx] = None
                        continue
                    H, W = img.shape[:2]
                    mast3r_w, mast3r_h = get_mast3r_image_shape(W, H)[:2]
                    K_anchor_512 = recompute_intrinsics(K_cam, (W, H), size=512)

                    image_idx = os.path.splitext(os.path.basename(anchor['image']))[0]
                    dpath = os.path.join(cfg['depth_root'], scene, f"{image_idx}.npy")
                    if not os.path.exists(dpath):
                        scene_maps[method][idx] = None
                        continue
                    d = np.load(dpath, mmap_mode='r').astype(np.float32)
                    d = cv2.resize(d, (mast3r_w, mast3r_h), interpolation=cv2.INTER_NEAREST)
                    scene_maps[method][idx] = generate_scene_map_from_depth(d, K_anchor_512)
                except Exception:
                    scene_maps[method][idx] = None

        # processed ledger (per scene)
        temp_file = processed_pairs_path(args.temp_file_prefix, scene)
        processed_pairs = load_processed_pairs(temp_file)

        # === Sequence-wise processing ===
        for sequence, seq_df in pairs_by_sequence.items():
            print(f"\n[Sequence] {scene}/{sequence}")

            # Load metadata for this sequence
            metadata_path = os.path.join(args.mapillary_root, scene, sequence, 'metadata.csv')
            if not os.path.exists(metadata_path):
                print(f"[!] Missing metadata.csv for {scene}/{sequence}, skipping this sequence.")
                continue
            meta_df = pd.read_csv(metadata_path)
            if 'id' not in meta_df.columns:
                print(f"[!] metadata.csv at {metadata_path} does not have 'id' column, skipping sequence.")
                continue
            meta_df['id'] = meta_df['id'].astype(str)
            meta_idx = {iid: row for iid, row in meta_df.set_index('id').iterrows()}

            # Output CSVs per sequence/method
            out_dir = os.path.join(args.output_root, scene, sequence)
            for method, cfg in METHOD_CONFIGS.items():
                out_file = os.path.join(out_dir, f"{cfg['filename_suffix']}.csv")
                ensure_dir_exists(out_file)
                create_csv_with_headers(out_file)

            # Cache query intrinsics per query_id within this sequence
            query_intrinsics_cache = {}  # qid -> (K_query_512 or None, dist_coeffs or None, q_img_path)

            # Iterate pairs in this sequence
            for _, row in tqdm(seq_df.iterrows(), total=len(seq_df), desc=f"{scene}/{sequence}"):
                anchor_idx = int(row['anchor_idx'])
                query_basename = os.path.splitext(os.path.basename(row['query_image']))[0]
                qid = query_basename

                # Skip if already processed (per-scene ledger with sequence)
                if (sequence, anchor_idx, qid) in processed_pairs:
                    continue

                # GPS ref from VBR at anchor index
                try:
                    ref_lat, ref_lon, compass_deg = extract_gps(traj_df, anchor_idx)
                except Exception:
                    # If GPS missing for this anchor, we still log failure for both methods
                    for method, cfg in METHOD_CONFIGS.items():
                        out_file = os.path.join(out_dir, f"{cfg['filename_suffix']}.csv")
                        save_failed_result_csv(out_file, sequence, qid, anchor_idx, method,
                                               np.nan, np.nan, np.nan, np.nan, np.nan)
                    mark_pair_processed(temp_file, sequence, anchor_idx, qid)
                    continue

                # Query intrinsics (cached)
                if qid not in query_intrinsics_cache:
                    meta = meta_idx.get(qid)
                    if meta is not None:
                        try:
                            f = float(meta.get('f', np.nan))
                        except Exception:
                            f = np.nan
                        try:
                            k1 = float(meta.get('k1', 0.0))
                            k2 = float(meta.get('k2', 0.0))
                            dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0])  # k1,k2,p1,p2,k3
                        except Exception:
                            dist_coeffs = None
                    else:
                        f, dist_coeffs = np.nan, None

                    q_img_path = os.path.join(args.mapillary_root, scene, sequence, f"{qid}.jpg")
                    K_query_512 = None
                    try:
                        q_img = cv2.imread(q_img_path)
                        if q_img is None:
                            raise FileNotFoundError(f"Missing query image: {q_img_path}")
                        Hq, Wq = q_img.shape[:2]
                        if f is None or not np.isfinite(f):
                            K_query_512 = None
                            dist_coeffs = None
                        else:
                            K_query = camera_matrix_from_mapillary_f(f, Wq, Hq)
                            K_query_512 = recompute_intrinsics(K_query, (Wq, Hq), size=512)
                    except Exception:
                        K_query_512, dist_coeffs = None, None
                    query_intrinsics_cache[qid] = (K_query_512, dist_coeffs, q_img_path)
                else:
                    K_query_512, dist_coeffs, q_img_path = query_intrinsics_cache[qid]

                # Load anchor sample
                try:
                    anchor = dataset[anchor_idx]
                    anchor['idx'] = anchor_idx
                except Exception:
                    for method, cfg in METHOD_CONFIGS.items():
                        out_file = os.path.join(out_dir, f"{cfg['filename_suffix']}.csv")
                        save_failed_result_csv(out_file, sequence, qid, anchor_idx, method,
                                               np.nan, np.nan, compass_deg, ref_lat, ref_lon)
                    mark_pair_processed(temp_file, sequence, anchor_idx, qid)
                    continue

                # Process each method
                for method, cfg in METHOD_CONFIGS.items():
                    out_file = os.path.join(out_dir, f"{cfg['filename_suffix']}.csv")
                    scene_map = None
                    if method == 'anydepth80':
                        scene_map = scene_maps[method].get(anchor_idx)

                    # If intrinsics missing, write failure and move on
                    if K_query_512 is None:
                        save_failed_result_csv(out_file, sequence, qid, anchor_idx, method,
                                               np.nan, np.nan, compass_deg, ref_lat, ref_lon)
                        continue

                    process_pair(
                        model, args.device, anchor, q_img_path, qid,
                        K_cam, T_base_cam, T_cam_lidar,
                        args.conf_percentile, args.min_inliers,
                        K_query_512, dist_coeffs, scene_map,
                        ref_lat, ref_lon, compass_deg,
                        out_file, method, sequence
                    )

                # mark processed once for both methods
                mark_pair_processed(temp_file, sequence, anchor_idx, qid)

        # cleanup
        del scene_maps
        import gc; gc.collect()
        print(f"[✓] Completed: {scene}")

if __name__ == "__main__":
    main()
