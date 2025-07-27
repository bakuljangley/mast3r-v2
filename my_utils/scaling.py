import numpy as np
from .mast3r_utils import solve_pnp


def compute_umeyama_similarity(A, B):
    """
    Estimate similarity transform (s, R, t) such that s*R*A + t ≈ B.
    Returns: scale, rotation matrix, translation vector
    """
    assert A.shape == B.shape
    n, m = A.shape

    mean_A = np.mean(A, axis=0)
    mean_B = np.mean(B, axis=0)
    AA = A - mean_A
    BB = B - mean_B

    # Compute covariance matrix
    H = AA.T @ BB / n
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_A = np.var(AA, axis=0).sum()
    if var_A < 1e-6:
        raise ValueError("Variance of A is zero, cannot compute similarity transform.")
    scale = np.sum(S) / var_A
    t = mean_B - scale * R @ mean_A
    return scale, R, t

def compute_scaled_points(method, master_pts, lidar_pts):
    """
    Compute scaled MASt3R points and the scale(s) used.

    Returns:
        scaled_pts (np.ndarray or None): Nx3 scaled points or None if failed
        scale (float or np.ndarray or None): scale factor(s) used, or None if failed
    """
    if len(master_pts) < 1 or len(lidar_pts) < 1 or master_pts.shape != lidar_pts.shape:
        return None, None

    try:
        if method in ['l1_translation', 'v1']:
            t_m = np.mean(master_pts, axis=0)
            t_l = np.mean(lidar_pts, axis=0)
            norm_m = np.linalg.norm(t_m, ord=1)
            norm_l = np.linalg.norm(t_l, ord=1)
            if norm_m < 1e-6:
                return None, None
            scale = norm_l / norm_m
            scaled_pts = master_pts * scale

        elif method in ['alignment', 'v2']:
            numerator = np.sum(master_pts * lidar_pts)
            denominator = np.sum(master_pts ** 2)
            if denominator < 1e-6:
                return None, None
            scale = numerator / denominator
            scaled_pts = master_pts * scale

        elif method in ['l1_centroid', 'v3']:
            norm_m = np.linalg.norm(master_pts, ord=2, axis=0)
            norm_l = np.linalg.norm(lidar_pts, ord=2, axis=0)
            c_m = np.mean(norm_m)
            c_l = np.mean(norm_l)
            if c_m < 1e-6:
                return None, None
            scale = c_l / c_m
            scaled_pts = master_pts * scale

        elif method in ['l1_axis', 'v4']:
            # Compute the norm (average distance to origin) per axis
            norm_m = np.mean(np.abs(master_pts), axis=0)
            norm_l = np.mean(np.abs(lidar_pts), axis=0)

            # Handle cases where master_pts has zero norm on an axis
            scale = np.zeros_like(norm_m)
            for i in range(3):
                if norm_m[i] < 1e-6:
                    return None, None  # Or another default value, like 0.0 or np.nan
                else:
                    scale[i] = norm_l[i] / norm_m[i]
            # Apply the per-axis scaling
            scaled_pts = master_pts * scale

        elif method in ['icp', 'umeyama']:
            scale, R, t = compute_umeyama_similarity(master_pts, lidar_pts)
            scaled_pts = scale * master_pts
            return scaled_pts, scale

        else:
            raise ValueError(f"Unknown scaling method: {method}")

        return scaled_pts, scale
    
    except Exception as e:
        print(f"Scaling failed for method {method}: {e}")
        return None, None

def scale_pnp(method, master_pts, lidar_pts, image_pts2d, K):
    """
    Apply scaling and solve PnP, returning SE(3) transformation or None.
    """
    scaled_pts, scale = compute_scaled_points(method, master_pts, lidar_pts)
    if scaled_pts is None:
        # Return nan arrays instead of None for consistent handling
        return None, np.array([np.nan]), None

    # Ensure scale is handled consistently
    if isinstance(scale, (list, np.ndarray)):  # For v4 and icp
        scale_values = np.array(scale)
    else:  # For v1, v2, v3 (scalar scale)
        scale_values = np.array([scale]) if scale is not None else np.array([np.nan])

    T = solve_pnp(scaled_pts, image_pts2d, K)
    return scaled_pts, scale_values, T