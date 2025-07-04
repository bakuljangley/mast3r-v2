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
    pose_to_se3, se3_to_pose, solve_pnp
)
from my_utils.scaling import scale_pnp

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
    return parser.parse_args()

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

    #load anchor query pairs 
    pairs_df = pd.read_csv(args.pairs_csv)

    # Initialize output containers
    mast3r_poses = []
    lidar_poses = []
    mast3r_scaled_v3_poses = []  # L1 centroid
    mast3r_scaled_v4_poses = []  # Per-axis L1
    mast3r_scaled_icp_poses = []  # ICP/Umeyama

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Estimating poses"):
        anchor_idx, query_idx = int(row['anchor_idx']), int(row['query_idx'])
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

            # PnP from MASt3R -- returned transformation is from anchor camera frame to query camera frame
            T_query_anchor_mast3r = solve_pnp(
                pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]],
                inlier_im1, K_new
            ) if len(inlier_im0) >= 4 else None

            # PnP from LiDAR
            T_query_anchor_lidar = solve_pnp(
                scene_map[valid_lidar_uv[:, 1], valid_lidar_uv[:, 0]],
                inlier_im1, K_new
            ) if len(valid_lidar_uv) >= 4 else None

            T_anchor_base = pose_to_se3(anchor['pose'])

            # query camera in local frame (mast3r pointmap)
            if T_query_anchor_mast3r is not None:
                T_m_local = T_anchor_base @ T_base_cam @ np.linalg.inv(T_query_anchor_mast3r)
                pose = se3_to_pose(T_m_local) #covert to pose
                mast3r_poses.append([query_idx, anchor_idx] + pose)

            # query camera in local frame (lidar pointcloud)
            if T_query_anchor_lidar is not None:
                T_l_local = T_anchor_base @ T_base_cam @ np.linalg.inv(T_query_anchor_lidar)
                pose = se3_to_pose(T_l_local) #convert to pose
                lidar_poses.append([query_idx, anchor_idx] + pose)

            # query camera in local frame (mast3r pointmap scaled -- v3 (L1 centroid scale)) 
            if len(inlier_im0) >= 1:
                master_pts = pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]]
                lidar_pts = scene_map[valid_lidar_uv[:, 1], valid_lidar_uv[:, 0]]
                T_scaled = scale_pnp('v3',master_pts, lidar_pts, inlier_im1, K_new)
                if T_scaled is not None:
                    T_s3_local = T_anchor_base @ T_base_cam @ np.linalg.inv(T_scaled)
                    pose = se3_to_pose(T_s3_local) #convert to pose
                    mast3r_scaled_v3_poses.append([query_idx, anchor_idx] + pose)

           # query camera in local frame (mast3r pointmap scaled -- v3 (L1 centroid scale per direction)) 
            if len(inlier_im0) >= 1:
                master_pts = pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]]
                lidar_pts = scene_map[valid_lidar_uv[:, 1], valid_lidar_uv[:, 0]]
                T_scaled = scale_pnp('v4',master_pts, lidar_pts, inlier_im1, K_new)
                if T_scaled is not None:
                    T_s4_local = T_anchor_base @ T_base_cam @ np.linalg.inv(T_scaled)
                    pose = se3_to_pose(T_s4_local)
                    mast3r_scaled_v4_poses.append([query_idx, anchor_idx] + pose)

            # query camera in local frame (mast3r pointmap scaled -- ICP/Umeyama)
            if len(inlier_im0) >= 1:
                master_pts = pts3d_im0[inlier_im0[:, 1], inlier_im0[:, 0]]
                lidar_pts = scene_map[valid_lidar_uv[:, 1], valid_lidar_uv[:, 0]]
                T_scaled = scale_pnp('icp', master_pts, lidar_pts, inlier_im1, K_new)
                if T_scaled is not None:
                    T_icp_local = T_anchor_base @ T_base_cam @ np.linalg.inv(T_scaled)
                    pose = se3_to_pose(T_icp_local)
                    mast3r_scaled_icp_poses.append([query_idx, anchor_idx] + pose)


    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


    np.savetxt(f"{args.output_prefix}_mast3r.txt", mast3r_poses, fmt="%.6f")
    np.savetxt(f"{args.output_prefix}_lidar.txt", lidar_poses, fmt="%.6f")
    np.savetxt(f"{args.output_prefix}_mast3r_scaled_v3.txt", mast3r_scaled_v3_poses, fmt="%.6f")
    np.savetxt(f"{args.output_prefix}_mast3r_scaled_v4.txt", mast3r_scaled_v4_poses, fmt="%.6f")
    np.savetxt(f"{args.output_prefix}_mast3r_scaled_icp.txt", mast3r_scaled_icp_poses, fmt="%.6f")
    print("Saved all pose estimates.")
    

if __name__ == "__main__":
    main()