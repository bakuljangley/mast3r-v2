import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
import cv2
from pyproj import CRS, Transformer
from pyproj.database import query_utm_crs_info
from pyproj.aoi import AreaOfInterest

from my_vbr_utils.vbr_dataset import vbrInterpolatedDataset, load_calibration, get_paths_from_scene
from my_vbr_utils.utilities import load_scene_correspondences
from my_utils.transformations import pose_to_se3

def localize_and_align_trajectory(
    vbr_dataset,
    calib,
    correspondence_json,
    verbose=True
):
    """
    Localize camera trajectory using correspondences and align local trajectory to global GPS.

    Parameters:
    -----------
    scene_name : str
        Name of the scene (used for calibration).
    vbr_dataset : VBRDataset
        Loaded VBR dataset object.
    calib : dict
        Camera calibration dictionary for the scene.
    correspondence_json : dict
        Dictionary loaded from correspondences JSON.
    verbose : bool
        If True, prints progress and intermediate results.

    Returns:
    --------
    dict with keys:
        - 'aligned_latlon_traj': list of (lat, lon) tuples for aligned trajectory
        - 'scale': estimated similarity scale
        - 'rotation': estimated similarity rotation matrix
        - 'translation': estimated similarity translation vector
        - 'camera_positions': list of dicts with per-image localization results
    """
    # --- Extract data from JSON ---
    image_indices = correspondence_json['image_indices']
    pixel_locations = correspondence_json['pixel_locations']
    locations = correspondence_json['locations']
    gps = correspondence_json['gps']
    
    # --- Find UTM zone ---
    lat, lon = gps
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=lon, east_lon_degree=lon,
            south_lat_degree=lat, north_lat_degree=lat
        )
    )
    if not utm_crs_list:
        raise ValueError("Could not find UTM CRS for GPS coordinates.")
    
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    wgs84_crs = CRS.from_epsg(4326)
    to_utm = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)
    to_latlon = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    # --- Camera Intrinsics ---
    K = calib['cam_l']['K'].astype(np.float32)
    dist = calib['cam_l']['dist_coeffs'].astype(np.float32)

    # --- Solve PnP for all images ---
    camera_positions = []
    for idx in range(len(image_indices)):
        pixels = np.array(pixel_locations[idx], dtype=np.float32)
        gps_coords = locations[idx]
        utm_coords = np.array([to_utm.transform(lon, lat) for lat, lon in gps_coords], dtype=np.float32)
        object_points = np.hstack([utm_coords, np.zeros((len(utm_coords), 1), dtype=np.float32)])

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, pixels, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            if verbose:
                print(f"PnP failed for index {idx}")
            continue
        # print(idx, tvec)
        R, _ = cv2.Rodrigues(rvec)
        camera_pos_utm = -R.T @ tvec
        lon, lat = to_latlon.transform(camera_pos_utm[0].item(), camera_pos_utm[1].item())

        pose = vbr_dataset[image_indices[idx]]['pose']
        T_utm_local_i = None
        if not (pose == -1).all():
            T_cam_utm = np.eye(4)
            T_cam_utm[:3, :3] = R
            T_cam_utm[:3, 3] = tvec.reshape(-1)
            T_utm_cam = np.linalg.inv(T_cam_utm)
            T_local_cam = pose_to_se3(pose)
            T_local_cam_inv = np.linalg.inv(T_local_cam)
            T_utm_local_i = T_utm_cam @ T_local_cam_inv

        camera_positions.append({
            'image_index': image_indices[idx],
            'latlon': [lat, lon],
            'R': R,
            'tvec': tvec,
            'T_utm_local': T_utm_local_i
        })

        if verbose:
            print(f"   Image {image_indices[idx]} Estimated Camera Position:")
            print(f"   Latitude : {lat:.7f}, Longitude: {lon:.7f}")

    if not camera_positions:
        raise RuntimeError("No successful PnP localizations.")

    # --- Similarity Transform ---
    global_latlon = [cp['latlon'] for cp in camera_positions]
    local_pts = []
    global_pts = []
    for cp in camera_positions:
        local_xyz = vbr_dataset[cp['image_index']]['pose'][:2]
        local_pts.append(local_xyz)
        lat, lon = cp['latlon']
        utm = np.array(to_utm.transform(lon, lat))
        global_pts.append(utm)
    local_pts = np.vstack(local_pts)
    global_pts = np.vstack(global_pts)

    # Compute similarity
    def compute_similarity(source, target):
        src_mean = source.mean(axis=0)
        tgt_mean = target.mean(axis=0)
        # print(source, target, "\n", src_mean, tgt_mean)
        src_c = source - src_mean
        tgt_c = target - tgt_mean
        scale = np.linalg.norm(tgt_c) / np.linalg.norm(src_c)
        H = src_c.T @ tgt_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = tgt_mean - scale * (R @ src_mean)
        return scale, R, t

    scale, rotation, translation = compute_similarity(local_pts, global_pts)

    # Apply to full trajectory
    local_traj_full = np.array(vbr_dataset.get_local_trajectory())
    aligned_utm = scale * (rotation @ local_traj_full[:, :2].T).T + translation
    xs, ys = aligned_utm[:, 0], aligned_utm[:, 1]
    lons, lats = to_latlon.transform(xs, ys)
    aligned_latlon_traj = list(zip(lats, lons))

    if verbose:
        print(f"\n Similarity Transform Parameters:")
        print(f"Scale: {scale:.6f}")
        print(f"Rotation:\n{rotation}")
        print(f"Translation: {translation}\n")

    # Extract yaw from similarity transform
    yaw = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))

    # Get local poses
    local_poses = np.array([vbr_dataset[i]['pose'] for i in range(len(vbr_dataset))])
    local_poses_se3 = np.array([pose_to_se3(pose) for pose in local_poses])

    # Compute global headings using numpy operations
    global_headings = []
    R_global_yaw = np.array([[np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0],
                             [np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0],
                             [0, 0, 1]])

    for pose in local_poses_se3:
        local_rotation = pose[:3, :3]
        global_rotation = R_global_yaw @ local_rotation
        global_heading = np.degrees(np.arctan2(global_rotation[1, 0], global_rotation[0, 0]))
        global_headings.append(global_heading)

    compass_headings = np.array(global_headings)

    return {
        'aligned_latlon_traj': aligned_latlon_traj,
        'scale': scale,
        'rotation': rotation,
        'translation': translation,
        'camera_positions': camera_positions,
        'compass_headings': compass_headings  # Add the compass headings to the results
    }

def save_trajectory(scene_name, latlon_traj, compass_headings, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'trajectory.csv')

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_index', 'latitude', 'longitude', 'heading'])
        for i, ((lat, lon), heading) in enumerate(zip(latlon_traj, compass_headings)):
            writer.writerow([i, lat, lon, heading])
    print(f"Saved trajectory to {csv_path}")

    # Optional: Plot
    lats, lons = zip(*latlon_traj)
    plt.figure(figsize=(10, 6))
    plt.plot(lons, lats, marker='o', linewidth=2)
    plt.title(f'Aligned Trajectory: {scene_name}')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'trajectory_plot.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', required=True, help="Scene name, e.g. 'campus_train0'")
    parser.add_argument('--dataset_root', required=True, help="Root of VBR dataset")
    parser.add_argument('--utils_root', required=True, help="Path to `my_vbr_utils/`")
    parser.add_argument('--save_dir', required=True, help="Directory to save outputs per scene")
    args = parser.parse_args()

    scene_name = args.scene
    dataset_root = args.dataset_root
    utils_root = args.utils_root
    save_dir = os.path.join(args.save_dir, scene_name)

    calib_path = get_paths_from_scene(dataset_root, scene_name)[-1]
    calib = load_calibration(calib_path)
    vbr_scene = vbrInterpolatedDataset(dataset_root, scene_name)

    correspondence_json = os.path.join(utils_root, "GPSalignment", f"{scene_name}.json")
    data = load_scene_correspondences(correspondence_json)

    results = localize_and_align_trajectory(vbr_scene, calib, data, verbose=True)
    save_trajectory(scene_name, results['aligned_latlon_traj'], results['compass_headings'], save_dir)

if __name__ == "__main__":
    main()
