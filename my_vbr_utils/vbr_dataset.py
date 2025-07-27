import os
import json
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

def get_paths_from_scene(root,scene):
    scene_name = scene.split("_")[0]
    img_dir = Path(root)/f"{scene_name}"/f"{scene}_kitti/"
    lidar_dir = Path(root)/f"{scene_name}"/f"{scene}_kitti/ouster_points"
    calib_path = Path (root)/f"{scene_name}"/f"{scene}"/f"vbr_calib.yaml"
    gt_poses = Path(root)/f"{scene_name}"/f"{scene}"/f"{scene}_gt.txt"
    return img_dir, gt_poses, lidar_dir, calib_path

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

    def __init__(self, dataset_root_dir, scene_name, max_time_diff=0.1, camera_mode = 'left',verbose=True):
        root_dir, gt_poses, lidar_dir, _ = get_paths_from_scene(dataset_root_dir,scene_name)
        self.root_dir      = root_dir
        self.max_time_diff = max_time_diff  # in seconds

        # ── 1) BUILD LIDAR DTYPE FROM metadata.json ─────────────────────────
        meta_path = lidar_dir/'.metadata.json'
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
        lidar_data_dir    = lidar_dir/'data'
        self.lidar_files = sorted(
            [str(fn) for fn in lidar_data_dir.glob('*.bin')]
        )
        if not self.lidar_files:
            raise RuntimeError(f"No .bin files found in {lidar_data_dir}")

        filesize = os.path.getsize(self.lidar_files[0])
        itemsize = minimal_sz
        while filesize % itemsize != 0:
            itemsize += 1

        self.lidar_dtype = np.dtype({
            'names': names,
            'formats': formats,
            'offsets': offsets,
            'itemsize': itemsize
        })

        # ── Load Timestamps ──
        ts_path = lidar_dir / 'timestamps.txt'
        self.lidar_timestamps = [
            pd.to_datetime(line).timestamp()
            for line in open(ts_path).read().splitlines()
        ]

        if len(self.lidar_timestamps) != len(self.lidar_files):
            raise ValueError(f"Mismatch: {len(self.lidar_timestamps)} lidar timestamps but {len(self.lidar_files)} lidar files")

        # ── Load Camera Data Based on Mode ──
        if camera_mode == 'left':
            img_ts_path = root_dir / 'camera_left' / 'timestamps.txt'
            img_data_dir = root_dir / 'camera_left' / 'data'
        elif camera_mode == 'right':
            img_ts_path = root_dir.parent.parent / 'camera_right' / 'timestamps.txt'
            img_data_dir = root_dir.parent.parent / 'camera_right' / 'data'
        else:
            raise ValueError(f"Invalid camera_mode: {camera_mode}. Must be 'left' or 'right'")

        # Load timestamps and files for selected camera
        self.image_timestamps = [
            pd.to_datetime(line).timestamp()
            for line in open(img_ts_path).read().splitlines()
        ]

        self.image_files = sorted(
            [str(fn) for fn in img_data_dir.glob('*.png')]
        )

        # ── VALIDATION ──
        if len(self.image_timestamps) != len(self.image_files):
            raise ValueError(f"Mismatch: {len(self.image_timestamps)} image timestamps but {len(self.image_files)} image files")
        
        if len(self.lidar_timestamps) != len(self.lidar_files):
            raise ValueError(f"Mismatch: {len(self.lidar_timestamps)} lidar timestamps but {len(self.lidar_files)} lidar files")

        # ── Load Poses ──
        self.poses = pd.read_csv(
            gt_poses,
            sep=r"\s+",
            comment='#',
            names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        )
        self.pose_timestamps = self.poses['timestamp'].values

        if verbose:
            print("Loaded vbrInterpolatedDataset from scene: ", scene_name)
            print("Images loaded from: ", img_data_dir)
            print("Ground Truth Poses: ", gt_poses)


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
            'image':  self.image_files[idx],
            'lidar_points': lidar_pts,
            'pose':         pose,
            'timestamp':    img_time
        }
    
    def get_lidar(self, idx):
        img_time = self.image_timestamps[idx]

        # — LiDAR —
        lidar_path = self.get_closest_lidar(img_time)
        if lidar_path is None:
            lidar_pts = np.array([[-1, -1, -1]])
        else:
            raw = np.fromfile(lidar_path, dtype=self.lidar_dtype)
            # extract x,y,z:
            lidar_pts = np.stack([raw['x'], raw['y'], raw['z']], axis=-1)
        return lidar_pts

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



class vbrInterpolatedDataset:
    ROS2NP = {
        1: 'i1',   # INT8
        2: 'u1',   # UINT8
        3: 'i2',   # INT16
        4: 'u2',   # UINT16
        5: 'i4',   # INT32
        6: 'u4',   # UINT32
        7: 'f4',   # FLOAT32
        8: 'f8',   # FLOAT64,
    }

    def __init__(self, dataset_root_dir, scene_name, max_time_diff=0.1, camera_mode='left', verbose=True):
        root_dir, gt_poses, lidar_dir, _ = get_paths_from_scene(dataset_root_dir,scene_name)
        self.root_dir = root_dir
        self.max_time_diff = max_time_diff
        self.camera_mode = camera_mode
        # ── Load LiDAR Metadata ──
        meta_path = lidar_dir/'.metadata.json'
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        fields = meta['fields']
        names = [fld['name'] for fld in fields]
        formats = [self.ROS2NP[fld['datatype']] for fld in fields]
        offsets = [fld['offset'] for fld in fields]

        # Calculate itemsize
        last_off = max(offsets)
        last_fmt = formats[offsets.index(last_off)]
        minimal_sz = last_off + np.dtype(last_fmt).itemsize

        lidar_data_dir = lidar_dir/'data'
        lidar_data_dir = lidar_dir / 'data'
        self.lidar_files = sorted(
            [str(fn) for fn in lidar_data_dir.glob('*.bin')]
        )
        if not self.lidar_files:
            raise RuntimeError(f"No .bin files found in {lidar_data_dir}")

        filesize = os.path.getsize(self.lidar_files[0])
        itemsize = minimal_sz
        while filesize % itemsize != 0:
            itemsize += 1

        self.lidar_dtype = np.dtype({
            'names': names,
            'formats': formats,
            'offsets': offsets,
            'itemsize': itemsize
        })

        # ── Load Timestamps ──
        ts_path = lidar_dir / 'timestamps.txt'
        self.lidar_timestamps = [
            pd.to_datetime(line).timestamp()
            for line in open(ts_path).read().splitlines()
        ]

        if len(self.lidar_timestamps) != len(self.lidar_files):
            raise ValueError(f"Mismatch: {len(self.lidar_timestamps)} lidar timestamps but {len(self.lidar_files)} lidar files")

        # ── Load Camera Data Based on Mode ──
        if camera_mode == 'left':
            img_ts_path = root_dir / 'camera_left' / 'timestamps.txt'
            img_data_dir = root_dir / 'camera_left' / 'data'
        elif camera_mode == 'right':
            img_ts_path = root_dir.parent.parent / 'camera_right' / 'timestamps.txt'
            img_data_dir = root_dir.parent.parent / 'camera_right' / 'data'
        else:
            raise ValueError(f"Invalid camera_mode: {camera_mode}. Must be 'left' or 'right'")

        # Load timestamps and files for selected camera
        self.image_timestamps = [
            pd.to_datetime(line).timestamp()
            for line in open(img_ts_path).read().splitlines()
        ]

        self.image_files = sorted(
            [str(fn) for fn in img_data_dir.glob('*.png')]
        )

        # ── VALIDATION ──
        if len(self.image_timestamps) != len(self.image_files):
            raise ValueError(f"Mismatch: {len(self.image_timestamps)} image timestamps but {len(self.image_files)} image files")
        
        if len(self.lidar_timestamps) != len(self.lidar_files):
            raise ValueError(f"Mismatch: {len(self.lidar_timestamps)} lidar timestamps but {len(self.lidar_files)} lidar files")

        # ── Load Poses ──
        self.poses = pd.read_csv(
            gt_poses,
            sep=r"\s+",
            comment='#',
            names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        )
        self.pose_timestamps = self.poses['timestamp'].values

        if verbose:
            print("Loaded vbrInterpolatedDataset from scene: ", scene_name)
            print("Images loaded from: ", img_data_dir)
            print("Ground Truth Poses: ", gt_poses)

    def __len__(self):
        ## Returns the number of images we have by timestamps
        return len(self.image_timestamps)

    def interpolate_pose(self, img_time):
        diffs = self.pose_timestamps - img_time
        idx_before = np.where(diffs <= 0, diffs, -np.inf).argmax()
        idx_after = np.where(diffs > 0, diffs, np.inf).argmin()

        if idx_before == -np.inf or idx_after == np.inf:
            return None

        t_before = self.pose_timestamps[idx_before]
        t_after = self.pose_timestamps[idx_after]
            # Handle case where timestamps are identical
        if t_before == t_after:
            # No interpolation needed, just return the pose
            return self.poses.iloc[idx_before][['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']].to_numpy()
        pose_before = self.poses.iloc[idx_before][['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']].to_numpy()
        pose_after = self.poses.iloc[idx_after][['tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']].to_numpy()

        alpha = (img_time - t_before) / (t_after - t_before)
        interpolated_translation = pose_before[:3] + alpha * (pose_after[:3] - pose_before[:3])
        interpolated_rotation = self.slerp(pose_before[3:], pose_after[3:], alpha)

        return np.concatenate([interpolated_translation, interpolated_rotation])

    def interpolate_lidar(self, img_time):
        # diffs = np.array(self.lidar_timestamps) - img_time
        # idx_before = np.where(diffs <= 0, diffs, -np.inf).argmax()
        # idx_after = np.where(diffs > 0, diffs, np.inf).argmin()
        # if idx_before == -np.inf or idx_after == np.inf:
        #     return None
        # t_before = self.lidar_timestamps[idx_before]
        # t_after = self.lidar_timestamps[idx_after]
        # # Handle case where timestamps are identical
        # if t_before == t_after:
        # # No interpolation needed, just return the lidar data
        #     lidar_data = np.fromfile(self.lidar_files[idx_before], dtype=self.lidar_dtype)
        #     return np.column_stack([lidar_data['x'], lidar_data['y'], lidar_data['z']])
        # # Read the structured LiDAR data correctly
        # lidar_before = np.fromfile(self.lidar_files[idx_before], dtype=self.lidar_dtype)
        # lidar_after = np.fromfile(self.lidar_files[idx_after], dtype=self.lidar_dtype)
        # alpha = (img_time - t_before) / (t_after - t_before)
        # # Extract x, y, z coordinates from structured arrays
        # xyz_before = np.column_stack([lidar_before['x'], lidar_before['y'], lidar_before['z']])
        # xyz_after = np.column_stack([lidar_after['x'], lidar_after['y'], lidar_after['z']])
        # # Interpolate the xyz coordinates
        # interpolated_xyz = xyz_before + alpha * (xyz_after - xyz_before)
        # return interpolated_xyz
        diffs = np.abs(np.array(self.lidar_timestamps) - img_time)
        idx   = diffs.argmin()
        if diffs[idx] > self.max_time_diff:
            return None
        # print(self.lidar_files[idx])
        # Read the structured LiDAR data correctly
        lidar_data = np.fromfile(self.lidar_files[idx], dtype=self.lidar_dtype)
        # Extract x, y, z coordinates from structured arrays
        xyz = np.stack([lidar_data['x'], lidar_data['y'], lidar_data['z']], axis=-1)
        return xyz

    def slerp(self, q1, q2, alpha):
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        dot_product = np.dot(q1, q2)

        if dot_product < 0.0:
            q2 = -q2
            dot_product = -dot_product

        if dot_product > 0.9995:
            return q1 + alpha * (q2 - q1)

        theta_0 = np.arccos(dot_product)
        theta = theta_0 * alpha
        q2_orthogonal = q2 - q1 * dot_product
        q2_orthogonal /= np.linalg.norm(q2_orthogonal)
        return q1 * np.cos(theta) + q2_orthogonal * np.sin(theta)

    def __getitem__(self, idx):
        ## loads item by image_timstamps
        img_time = self.image_timestamps[idx]

        lidar_pts = self.interpolate_lidar(img_time)
        if lidar_pts is None:
            lidar_pts = np.array([[-1, -1, -1]])

        pose = self.interpolate_pose(img_time)
        if pose is None:
            pose = np.array([-1, -1, -1, -1, -1, -1, -1])

        return {
            'image': self.image_files[idx],
            'lidar_points': lidar_pts,
            'pose': pose,
            'timestamp': img_time
        }
    
    def get_local_trajectory(self):
        """
        Returns an array of shape (N, 7) with [tx, ty, tz, qx, qy, qz, qw] for each image timestamp.
        """
        trajectory = []
        for img_time in self.image_timestamps:
            pose = self.interpolate_pose(img_time)
            if pose is not None and (pose != -1).all():
                trajectory.append(pose)
        return np.array(trajectory)
    
    def get_lidar(self,idx):
        ## loads item by image_timstamps
        img_time = self.image_timestamps[idx]
        lidar_pts = self.interpolate_lidar(img_time)
        if lidar_pts is None:
            lidar_pts = np.array([[-1, -1, -1]])

        return lidar_pts
    
    def get_pose(self,idx):
        ## loads item by image_timstamps
        img_time = self.image_timestamps[idx]
        pose = self.interpolate_pose(img_time)
        if pose is None:
            pose = np.array([-1, -1, -1, -1, -1, -1, -1])
        return pose
    

class CombinedDataset:
    def __init__(self, scene_names, root_dir, max_time_diff=0.1, camera_mode='left'):
        """
        Initialize the CombinedDataset with a list of scene names and a root directory.
        """
        self.datasets = [
            vbrInterpolatedDataset(root_dir, scene_name, max_time_diff, camera_mode)
            for scene_name in scene_names
        ]
        self.lengths = [len(dataset) for dataset in self.datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)

    def __len__(self):
        """Return the total number of samples across all datasets."""
        return self.total_length

    def __getitem__(self, idx):
        """
        Retrieve a sample from the combined datasets based on the global index.
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range for CombinedDataset with {self.total_length} samples")
        
        # Find which dataset this index belongs to
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        
        return self.datasets[dataset_idx][local_idx]
    
    def get_local_trajectory(self):
        """
        Returns a combined local trajectory for all datasets.
        Each entry is [tx, ty, tz, qx, qy, qz, qw].
        """
        trajs = [dataset.get_local_trajectory() for dataset in self.datasets]
        if trajs:
            return np.vstack(trajs)
        else:
            return np.empty((0, 7))

    def get_dataset_info(self):
        """Return information about the combined datasets."""
        info = []
        for i, dataset in enumerate(self.datasets):
            info.append({
                'dataset_index': i,
                'length': len(dataset),
                'start_index': self.cumulative_lengths[i],
                'end_index': self.cumulative_lengths[i+1] - 1
            })
        return info
    
    def get_lidar(self, idx):
        """
        Return lidar points for the given global index.
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range for CombinedDataset with {self.total_length} samples")
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx].get_lidar(local_idx)

    def get_pose(self, idx):
        """
        Return pose for the given global index.
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range for CombinedDataset with {self.total_length} samples")
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        return self.datasets[dataset_idx].get_pose(local_idx)
    
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