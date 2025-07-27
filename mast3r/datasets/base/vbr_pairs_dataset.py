from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from scipy.spatial.transform import Rotation as R
from .mast3r_base_stereo_view_dataset import MASt3RBaseStereoViewDataset


def get_paths_from_scene(scene, img_root, depth_root, pose_root):
    depth_dir = Path(depth_root)/f"{scene}"
    scene_name = scene.split("_")[0]
    img_dir = Path(img_root)/f"{scene_name}"/f"{scene}_kitti/camera_left/data"
    pose_dir = Path(pose_root)/f"{scene}.txt" #poses_path
    calib_path = Path (img_root)/f"{scene_name}"/f"{scene}"/f"vbr_calib.yaml"
    return img_dir, depth_dir, pose_dir, calib_path


class VBRPairsDataset(MASt3RBaseStereoViewDataset):
    def __init__(self, root_dir, scene, split, pairs_dir, depth_dir, pose_dir, **kwargs):
        super().__init__(**kwargs)
        self.image_dir, self.depth_dir, self.poses_txt, self.calib_path = get_paths_from_scene(scene, root_dir, depth_dir, pose_dir)
        self.calib = self._load_calibration(self.calib_path)
        self.pose_values = np.loadtxt(self.poses_txt, comments="#")
        self.pairs = self._load_pairs(pairs_dir, scene, split)
        self.num_views = 2
        self.is_metric_scale = True #overwrite the metric scale flag

    def __len__(self):
        return len(self.pairs)

    def _get_views(self, idx, resolution, rng):
        anchor_idx, query_idx = self.pairs[idx]
        return [self._view(anchor_idx), self._view(query_idx)]
    
    def _load_pairs(self, pairs_dir, scene, split): 
        """
        Load pairs based on the scene and split.
        """
        pairs_path = Path(pairs_dir) / f"{scene}" /f"{split}_pairs.txt"
        print(pairs_path)
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file for split '{split}' not found: {pairs_path}")
        return np.loadtxt(pairs_path, dtype=int)


    def _view(self, idx):
        img_path = self.image_dir / f"{idx:010d}.png"
        depth_path = self.depth_dir / f"{idx:010d}.npy"
        img = np.array(Image.open(img_path).convert("RGB"))
        depth = np.load(depth_path).astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        T_cam_lidar = self.calib['cam_l']['T_cam_lidar'].astype(np.float32)
        pose = np.linalg.inv(T_cam_lidar @ np.linalg.inv(self._pose(idx)))
        K = self.calib['cam_l']['K'].astype(np.float32)

        return {
            'img': img,
            'depthmap': depth,
            'camera_intrinsics': K,
            'camera_pose': pose.astype(np.float32),
            'dataset': 'vbr',
            'label': f'vbr_{idx:010d}',
            'instance': str(idx),
        }

    def _pose(self, idx):
        pose_vec = self.pose_values[idx]  # [tx, ty, tz, qx, qy, qz, qw]
        t = pose_vec[:3]
        q = pose_vec[3:]
        R_mat = R.from_quat(q).as_matrix()
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T

    def _load_calibration(self, yaml_path):
        return load_calibration(yaml_path)


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


def pose_vec_to_se3(pose):
    t = pose[:3]
    q = pose[3:]
    R_mat = R.from_quat(q).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3,  3] = t
    return T