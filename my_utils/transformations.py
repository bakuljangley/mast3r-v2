import numpy as np
from scipy.spatial.transform import Rotation as Rscipy
import cv2

def quat_to_rot(q):
    """Quaternion to rotation matrix"""
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ])

def pnp_to_se3(rvec, tvec):
    """Convert OpenCV PnP output (rvec, tvec) to a 4x4 SE(3) transformation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.ravel()
    return T

def pose_to_se3(pose_array):
    """Convert a pose [tx, ty, tz, qx, qy, qz, qw] to a 4x4 SE(3) transformation matrix."""
    tx, ty, tz, qx, qy, qz, qw = pose_array
    T = np.eye(4)
    T[:3, :3] = Rscipy.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = [tx, ty, tz]
    return T

def se3_to_pose(T):
    """
    Convert a 4x4 transformation SE(3) matrix to (tx, ty, tz, qx, qy, qz, qw).

    Args:
        T (np.ndarray): 4x4 transformation matrix.

    Returns:
        list: [tx, ty, tz, qx, qy, qz, qw]
    """
    t = T[:3, 3]
    q = Rscipy.from_matrix(T[:3, :3]).as_quat()
    return list(t) + list(q)