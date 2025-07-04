import os
import json
import numpy as np
import pandas as pd
import yaml


class vbrDataset:
    # ROS2 PointField datatype code → NumPy dtype string
    ROS2NP = {
        1: 'i1',   # INT8
        2: 'u1',   # UINT8
        3: 'i2',   # INT16
        4: 'u2',   # UINT16
        5: 'i4',   # INT32
        6: 'u4',   # UINT32
        7: 'f4',   # FLOAT32
        8: 'f8',   # FLOAT64
    }

    def __init__(self, root_dir, pose_file, max_time_diff=0.1):
        self.root_dir      = root_dir
        self.max_time_diff = max_time_diff  # in seconds

        # ── 1) BUILD LIDAR DTYPE FROM metadata.json ─────────────────────────
        meta_path = os.path.join(root_dir, 'ouster_points', '.metadata.json')
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        fields  = meta['fields']
        names   = [fld['name']             for fld in fields]
        formats = [self.ROS2NP[fld['datatype']] for fld in fields]
        offsets = [fld['offset']           for fld in fields]

        # minimal itemsize = last offset + its dtype size
        last_off   = max(offsets)
        last_fmt   = formats[offsets.index(last_off)]
        minimal_sz = last_off + np.dtype(last_fmt).itemsize

        # list all .bin files so we can auto-detect true point_step
        lidar_data_dir    = os.path.join(root_dir, 'ouster_points', 'data')
        self.lidar_files  = sorted(
            os.path.join(lidar_data_dir, fn)
            for fn in os.listdir(lidar_data_dir) if fn.endswith('.bin')
        )
        if not self.lidar_files:
            raise RuntimeError(f"No .bin files found in {lidar_data_dir}")

        # inspect the first file’s size, bump itemsize until it divides evenly
        filesize = os.path.getsize(self.lidar_files[0])
        itemsize = minimal_sz
        while filesize % itemsize != 0:
            itemsize += 1

        # final structured dtype
        self.lidar_dtype = np.dtype({
            'names':    names,
            'formats':  formats,
            'offsets':  offsets,
            'itemsize': itemsize
        })


        # ── 2) LOAD LiDAR TIMESTAMPS ────────────────────────────────────────
        ts_path = os.path.join(root_dir, 'ouster_points', 'timestamps.txt')
        self.lidar_timestamps = [
            pd.to_datetime(line).timestamp()
            for line in open(ts_path).read().splitlines()
        ]

        # ── 3) LOAD IMAGE TIMESTAMPS ────────────────────────────────────────
        img_ts = os.path.join(root_dir, 'camera_left', 'timestamps.txt')
        self.image_timestamps = [
            pd.to_datetime(line).timestamp()
            for line in open(img_ts).read().splitlines()
        ]
        # Left image paths
        img_left_dir = os.path.join(root_dir, 'camera_left', 'data')
        self.image_files_left = sorted(
            [os.path.join(img_left_dir, fn) for fn in os.listdir(img_left_dir) if fn.endswith('.png')],
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )


        # Right image paths
        img_right_dir = os.path.join(root_dir, 'camera_right', 'data')
        self.image_files_right = sorted(
            [os.path.join(img_right_dir, fn) for fn in os.listdir(img_right_dir) if fn.endswith('.png')],
            key=lambda x: int(os.path.basename(x).split('.')[0])
)
        # ── 4) LOAD POSES ───────────────────────────────────────────────────
        self.poses = pd.read_csv(
            pose_file,
            sep=r"\s+",
            comment='#',
            names=['timestamp','tx','ty','tz','qx','qy','qz','qw']
        )
        self.pose_timestamps = self.poses['timestamp'].values


    def __len__(self):
        return len(self.image_timestamps)


    def get_closest_lidar(self, img_time): #nearest neighbour search
        diffs = np.abs(np.array(self.lidar_timestamps) - img_time)
        idx   = diffs.argmin()
        if diffs[idx] > self.max_time_diff:
            return None
        return self.lidar_files[idx]


    def get_closest_pose(self, img_time):
        diffs = np.abs(self.pose_timestamps - img_time)
        idx   = diffs.argmin()
        if diffs[idx] > self.max_time_diff:
            return None
        row = self.poses.iloc[idx]
        return row[['tx','ty','tz','qx','qy','qz','qw']].to_numpy()


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))] #to account for slicing 
        img_time = self.image_timestamps[idx]

        # — LiDAR —
        lidar_path = self.get_closest_lidar(img_time)
        if lidar_path is None:
            lidar_pts = np.array([[-1, -1, -1]])
        else:
            raw = np.fromfile(lidar_path, dtype=self.lidar_dtype)
            # extract x,y,z:
            lidar_pts = np.stack([raw['x'], raw['y'], raw['z']], axis=-1)

        # — Pose —
        pose = self.get_closest_pose(img_time)
        if pose is None:
            pose = np.array([-1, -1, -1, -1, -1, -1, -1])

        return {
            'image_left':  self.image_files_left[idx],
            'image_right': self.image_files_right[idx],
            'lidar_points': lidar_pts,
            'pose':         pose,
            'timestamp':    img_time
        }


    def get_image_lidar_pose_stats(self):
        with_lidar = []
        with_pose  = []
        with_both  = []

        for i in range(len(self)):
            has_l = (self.get_closest_lidar(self.image_timestamps[i]) is not None)
            has_p = (self.get_closest_pose(self.image_timestamps[i]) is not None)
            if has_l: with_lidar.append(i)
            if has_p: with_pose.append(i)
            if has_l and has_p:
                with_both.append(i)

        return {
            "count_with_lidar": len(with_lidar),
            "images_with_lidar": with_lidar,
            "count_with_pose":  len(with_pose),
            "images_with_pose": with_pose,
            "count_with_both":  len(with_both),
            "images_with_both": with_both,
        }
    def get_local_trajectory_array(self):
        """
        Returns a NumPy array of shape (N_valid, 8), where each row is:
        [image_index, tx, ty, tz, qx, qy, qz, qw] — only for images that have a valid pose.
        """
        matched = []
        pose_ts = self.pose_timestamps
        pose_all = self.poses[['tx','ty','tz','qx','qy','qz','qw']].values

        for i, ts in enumerate(self.image_timestamps):
            diffs = np.abs(pose_ts - ts)
            j = diffs.argmin()
            if diffs[j] <= self.max_time_diff:
                matched.append([i, *pose_all[j]])

        return np.array(matched)



import yaml
import numpy as np

def load_calibration(yaml_path):
    """
    Load sensor calibration from YAML and return:

    calib['lidar']['T_base_lidar']   
    calib['imu']['T_base_imu']       

    For each camera 'cam_l' / 'cam_r':
      calib[cam]['K']                
      calib[cam]['dist_coeffs']      
      calib[cam]['resolution']      
      calib[cam]['T_base_cam']      
      calib[cam]['T_cam_lidar']      # LIDAR → CAMERA
      calib[cam]['T_lidar_cam']      
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    calib = {}

    # LIDAR → BASE
    T_base_lidar = np.array(data['lidar']['T_b'], dtype=float)
    calib['lidar'] = {
        'T_base_lidar': T_base_lidar
    }

    # IMU → BASE
    T_base_imu = np.array(data['os_imu']['T_b'], dtype=float)
    calib['imu'] = {
        'T_base_imu': T_base_imu
    }

    # CAMERAS
    for cam in ['cam_l', 'cam_r']:
        entry = data[cam]

        # Intrinsics
        fx, fy, cx, cy = entry['intrinsics']
        K = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1],
        ], dtype=float)

        # Distortion (radtan: [k1, k2, p1, p2])
        dist = np.array(entry['distortion_coeffs'], dtype=float)

        # Resolution
        w, h = entry['resolution']

        # BASE ← CAMERA
        T_base_cam = np.array(entry['T_b'], dtype=float)

        # LIDAR → CAMERA = inv(T_base_cam) @ T_base_lidar
        # so that X_cam = T_cam_lidar @ X_lidar
        T_cam_lidar = np.linalg.inv(T_base_cam) @ T_base_lidar

        # inverse, if needed
        T_lidar_cam = np.linalg.inv(T_cam_lidar)

        calib[cam] = {
            'K':             K,
            'dist_coeffs':   dist,
            'resolution':    (w, h),
            'T_base_cam':    T_base_cam,
            'T_cam_lidar':   T_cam_lidar,
            'T_lidar_cam':   T_lidar_cam,
        }

    return calib


def generate_depth_and_scene_maps(pts_lidar, K, T_cam_lidar, img_shape):
    """
    Returns:
    - depth_map: 2D array of shape (H, W) with depth values (np.inf for unobserved pixels)
    - scene_map: 3D array of shape (H, W, 3) with 3D points (np.nan for unobserved pixels)
    Only stores closest point per pixel (Z-buffer).
    """
    H, W = img_shape
    
    # Transform lidar → cam
    pts_hom = np.hstack([pts_lidar, np.ones((pts_lidar.shape[0], 1))])
    cam_pts = (T_cam_lidar @ pts_hom.T).T
    Xc, Yc, Zc = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]
    
    # Filter points in front of camera
    valid = Zc > 0
    Xc, Yc, Zc = Xc[valid], Yc[valid], Zc[valid]
    cam_pts = cam_pts[valid, :3]
    
    # Project to pixels
    u = np.round(K[0, 0] * Xc / Zc + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * Yc / Zc + K[1, 2]).astype(int)
    
    # Filter valid image coordinates
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, Zc, cam_pts = u[in_bounds], v[in_bounds], Zc[in_bounds], cam_pts[in_bounds]
    
    # Initialize output arrays
    depth_map = np.full((H, W), np.inf)
    scene_map = np.full((H, W, 3), np.nan)
    
    # Z-buffer update
    for i in range(len(u)):
        row, col = v[i], u[i]  # Image coordinates: row=v, column=u
        depth_val = Zc[i]
        if depth_val < depth_map[row, col]:
            depth_map[row, col] = depth_val
            scene_map[row, col] = cam_pts[i]
    
    return depth_map, scene_map

def overlap(matches, depth_map, max_pixel_dist=2):
    """
    For each (u, v) in matches, find the nearest valid depth pixel within max_pixel_dist.
    Returns:
        matched_uv:        array of (u, v) from matches that have a valid depth neighbor
        matched_lidar_uv:  array of (u, v) of the nearest valid depth pixel
        matched_indices:   indices of matches that were matched (for correspondence)
    """
    valid_mask = np.isfinite(depth_map)
    valid_v, valid_u = np.where(valid_mask)
    valid_uv = np.stack((valid_u, valid_v), axis=-1)
    matched_uv = []
    matched_lidar_uv = []
    matched_indices = []
    for idx, uv in enumerate(matches):
        u, v = int(round(uv[0])), int(round(uv[1]))
        dists = np.linalg.norm(valid_uv - [u, v], axis=1)
        within = np.where(dists <= max_pixel_dist)[0]
        if len(within) > 0:
            nearest_idx = within[np.argmin(dists[within])]
            nearest_u, nearest_v = valid_uv[nearest_idx]
            matched_uv.append([u, v])
            matched_lidar_uv.append([nearest_u, nearest_v])
            matched_indices.append(idx)
    return np.array(matched_uv), np.array(matched_lidar_uv), np.array(matched_indices)
