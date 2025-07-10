import os
import json
import numpy as np
import pandas as pd
import yaml

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

    def __init__(self, root_dir, pose_file, max_time_diff=0.1, camera_mode='left'):
        self.root_dir = root_dir
        self.max_time_diff = max_time_diff
        self.camera_mode = camera_mode
        # ── Load LiDAR Metadata ──
        meta_path = os.path.join(root_dir, 'ouster_points', '.metadata.json')
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

        lidar_data_dir = os.path.join(root_dir, 'ouster_points', 'data')
        self.lidar_files = sorted(
            os.path.join(lidar_data_dir, fn)
            for fn in os.listdir(lidar_data_dir) if fn.endswith('.bin')
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
        ts_path = os.path.join(root_dir, 'ouster_points', 'timestamps.txt')
        self.lidar_timestamps = [
            pd.to_datetime(line).timestamp()
            for line in open(ts_path).read().splitlines()
        ]

        if len(self.lidar_timestamps) != len(self.lidar_files):
            raise ValueError(f"Mismatch: {len(self.lidar_timestamps)} lidar timestamps but {len(self.lidar_files)} lidar files")

        # ── Load Camera Data Based on Mode ──
        if camera_mode == 'left':
            img_ts_path = os.path.join(root_dir, 'camera_left', 'timestamps.txt')
            img_data_dir = os.path.join(root_dir, 'camera_left', 'data')
        elif camera_mode == 'right':
            img_ts_path = os.path.join(root_dir, 'camera_right', 'timestamps.txt')
            img_data_dir = os.path.join(root_dir, 'camera_right', 'data')
        else:
            raise ValueError(f"Invalid camera_mode: {camera_mode}. Must be 'left' or 'right'")

        # Load timestamps and files for selected camera
        self.image_timestamps = [
            pd.to_datetime(line).timestamp()
            for line in open(img_ts_path).read().splitlines()
        ]

        self.image_files = sorted(
            [os.path.join(img_data_dir, fn) for fn in os.listdir(img_data_dir) if fn.endswith('.png')],
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )

        # ── VALIDATION ──
        if len(self.image_timestamps) != len(self.image_files):
            raise ValueError(f"Mismatch: {len(self.image_timestamps)} image timestamps but {len(self.image_files)} image files")
        
        if len(self.lidar_timestamps) != len(self.lidar_files):
            raise ValueError(f"Mismatch: {len(self.lidar_timestamps)} lidar timestamps but {len(self.lidar_files)} lidar files")

        # ── Load Poses ──
        self.poses = pd.read_csv(
            pose_file,
            sep=r"\s+",
            comment='#',
            names=['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        )
        self.pose_timestamps = self.poses['timestamp'].values

        # ── Print dataset statistics for verification ──
        # print(f"Dataset loaded successfully:")
        # print(f"  - Images: {len(self.image_timestamps)}")
        # print(f"  - LiDAR scans: {len(self.lidar_timestamps)}")
        # print(f"  - Poses: {len(self.poses)}")

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
        diffs = np.array(self.lidar_timestamps) - img_time
        idx_before = np.where(diffs <= 0, diffs, -np.inf).argmax()
        idx_after = np.where(diffs > 0, diffs, np.inf).argmin()
        if idx_before == -np.inf or idx_after == np.inf:
            return None
        t_before = self.lidar_timestamps[idx_before]
        t_after = self.lidar_timestamps[idx_after]
        # Handle case where timestamps are identical
        if t_before == t_after:
        # No interpolation needed, just return the lidar data
            lidar_data = np.fromfile(self.lidar_files[idx_before], dtype=self.lidar_dtype)
            return np.column_stack([lidar_data['x'], lidar_data['y'], lidar_data['z']])
        # Read the structured LiDAR data correctly
        lidar_before = np.fromfile(self.lidar_files[idx_before], dtype=self.lidar_dtype)
        lidar_after = np.fromfile(self.lidar_files[idx_after], dtype=self.lidar_dtype)
        alpha = (img_time - t_before) / (t_after - t_before)
        # Extract x, y, z coordinates from structured arrays
        xyz_before = np.column_stack([lidar_before['x'], lidar_before['y'], lidar_before['z']])
        xyz_after = np.column_stack([lidar_after['x'], lidar_after['y'], lidar_after['z']])
        # Interpolate the xyz coordinates
        interpolated_xyz = xyz_before + alpha * (xyz_after - xyz_before)
        
        return interpolated_xyz
        

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
    def __init__(self, datasets):
        """
        Initialize the CombinedDataset with a list of vbrInterpolatedDataset instances.
        Args:
            datasets: List of vbrInterpolatedDataset instances to combine.
        """
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = sum(self.lengths)

    def __len__(self):
        """Return the total number of samples across all datasets."""
        return self.total_length

    def __getitem__(self, idx):
        """
        Retrieve a sample from the combined datasets based on the global index.
        Args:
            idx: Global index across all datasets.
        Returns:
            Sample dictionary from the corresponding dataset.
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



class VBRDataset:
    def __init__(self, config_path, root_dir=None, scenes=None, locations=None, load_all=False):
        """
        Initialize VBR dataset manager with flexible loading options.
        
        Args:
            config_path: Path to the YAML configuration file
            root_dir: Root directory (overrides the one in config if provided)
            scenes: List of specific scene names to load (e.g., ['campus_train0', 'spagna_train0'])
            locations: List of locations to load all scenes from (e.g., ['campus', 'spagna'])
            load_all: If True, load all available scenes (default: False)
            
        Loading Priority:
            1. If scenes is specified, load only those scenes
            2. If locations is specified, load all scenes from those locations
            3. If load_all is True, load all available scenes
            4. If none specified, load nothing (empty dataset)
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Use provided root_dir or fall back to config
        self.root_dir = root_dir or self.config['vbr_datasets']['base_path']
        
        # Initialize storage
        self.datasets = {}
        self.scene_names = []
        
        # Determine what to load
        scenes_to_load = self._determine_scenes_to_load(scenes, locations, load_all)
        
        if not scenes_to_load:
            print("No scenes specified to load. Use scenes=[], locations=[], or load_all=True")
            return
        
        # Load the determined scenes
        self._load_scenes(scenes_to_load)
        
        print(f"Loaded {len(self.datasets)} training scenes: {self.scene_names}")
    
    def _determine_scenes_to_load(self, scenes, locations, load_all):
        """Determine which scenes to load based on parameters."""
        vbr_locations = {k: v for k, v in self.config['vbr_datasets'].items() if k != 'base_path'}
        
        # Get all available scenes from config
        all_available_scenes = []
        for location, splits in vbr_locations.items():
            if 'train' in splits:
                for dataset_info in splits['train']:
                    all_available_scenes.append({
                        'name': dataset_info['name'],
                        'location': location,
                        'kitti_path': dataset_info['kitti_path'],
                        'gt_path': dataset_info['gt_path']
                    })
        
        # Priority 1: Specific scenes
        if scenes is not None:
            scenes_to_load = []
            for scene_name in scenes:
                scene_info = next((s for s in all_available_scenes if s['name'] == scene_name), None)
                if scene_info:
                    scenes_to_load.append(scene_info)
                else:
                    print(f"Warning: Scene '{scene_name}' not found in config")
            return scenes_to_load
        
        # Priority 2: Specific locations
        if locations is not None:
            scenes_to_load = []
            for location in locations:
                location_scenes = [s for s in all_available_scenes if s['location'] == location]
                if location_scenes:
                    scenes_to_load.extend(location_scenes)
                else:
                    print(f"Warning: No scenes found for location '{location}'")
            return scenes_to_load
        
        # Priority 3: Load all
        if load_all:
            return all_available_scenes
        
        # Priority 4: Load nothing
        return []
    
    def _load_scenes(self, scenes_to_load):
        """Load the specified scenes."""
        for scene_info in scenes_to_load:
            scene_name = scene_info['name']
            kitti_path = os.path.join(self.root_dir, scene_info['kitti_path'])
            gt_path = os.path.join(self.root_dir, scene_info['gt_path'])
            
            print(f"Loading scene: {scene_name}")
            print(f"  KITTI path: {kitti_path}")
            print(f"  GT path: {gt_path}")
            
            try:
                # Load the dataset
                self.datasets[scene_name] = vbrInterpolatedDataset(kitti_path, gt_path)
                self.scene_names.append(scene_name)
                print(f"  ✓ Successfully loaded {scene_name}")
            except Exception as e:
                print(f"  ✗ Failed to load {scene_name}: {e}")
                continue  # Skip this dataset and continue with others
    
    def __getitem__(self, scene_name):
        """
        Get a specific scene dataset.
        Args:
            scene_name: Name of the scene (e.g., 'campus_train0', 'spagna_train0')
        Returns:
            vbrInterpolatedDataset instance
        """
        if scene_name not in self.datasets:
            raise KeyError(f"Scene '{scene_name}' not found. Available scenes: {self.scene_names}")
        return self.datasets[scene_name]
    
    def get_combined_dataset(self, scene_names=None):
        """
        Get a combined dataset with specified scenes.
        Args:
            scene_names: List of scene names to combine (None for all loaded scenes)
        Returns:
            CombinedDataset instance
        """
        if scene_names is None:
            scene_names = self.scene_names
        
        # Validate scene names
        for scene_name in scene_names:
            if scene_name not in self.datasets:
                raise KeyError(f"Scene '{scene_name}' not found. Available scenes: {self.scene_names}")
        
        datasets = [self.datasets[scene_name] for scene_name in scene_names]
        return CombinedDataset(datasets)
    
    def get_location_dataset(self, location):
        """
        Get combined dataset for all loaded scenes from a specific location.
        Args:
            location: Location name (e.g., 'campus', 'spagna')
        Returns:
            CombinedDataset instance
        """
        location_scenes = [name for name in self.scene_names if name.startswith(location)]
        
        if not location_scenes:
            raise ValueError(f"No loaded scenes found for location '{location}'")
        
        return self.get_combined_dataset(location_scenes)
    
    def list_scenes(self):
        """List all loaded scene names."""
        return self.scene_names.copy()
    
    def list_locations(self):
        """List all locations from loaded scenes."""
        locations = set()
        for scene_name in self.scene_names:
            location = scene_name.split('_')[0]
            locations.add(location)
        return sorted(list(locations))
    
    def __len__(self):
        """Return total number of loaded scenes."""
        return len(self.datasets)
    
    def __repr__(self):
        return f"VBRDataset({len(self.datasets)} scenes: {self.scene_names})"
    

def load_scene_calibration(scene_name, config_path):
    """
    Loads and returns the calibration dictionary for the given scene.
    Args:
        scene_name: e.g. 'spagna_train0'
        config_path: path to vbrPaths.yaml
    Returns:
        calib: dict loaded from vbr_calib.yaml
    """
    # Load the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_path = config['vbr_datasets']['base_path']
    # Find the location and split for the scene
    for location, splits in config['vbr_datasets'].items():
        if location == 'base_path':
            continue
        for split_type in ['train', 'test']:
            if split_type in splits:
                for split in splits[split_type]:
                    if split['name'] == scene_name:
                        calib_path = os.path.join(
                            base_path,
                            location,
                            scene_name,
                            "vbr_calib.yaml"
                        )
                        return load_calibration(calib_path)
    raise ValueError(f"Scene {scene_name} not found in config.")

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