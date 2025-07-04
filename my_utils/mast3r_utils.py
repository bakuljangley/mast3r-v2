import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rscipy
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import load_images
import time

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


def solve_pnp(pts3d, pts2d, K):
    """
    Solve PnP given matched 3D-2D correspondences.

    Args:
        pts3d (np.ndarray): Nx3 array of 3D points.
        pts2d (np.ndarray): Nx2 array of 2D pixel coordinates.
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        np.ndarray or None: 4x4 SE(3) transformation matrix, or None if PnP fails.
    """
    if len(pts3d) < 4 or len(pts2d) < 4:
        return None
    success, rvec, tvec, _ = cv2.solvePnPRansac(pts3d.astype(np.float32), pts2d.astype(np.float32), K, None)
    return pnp_to_se3(rvec, tvec) if success else None


def scale_intrinsics(K, prev_w, prev_h, target_w, target_h):
    """Scale the intrinsics matrix for new image dimensions."""
    assert K.shape == (3, 3), f"Expected (3, 3), got {K.shape=}"
    scale_w = target_w / prev_w
    scale_h = target_h / prev_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_w
    K_scaled[0, 2] *= scale_w
    K_scaled[1, 1] *= scale_h
    K_scaled[1, 2] *= scale_h
    return K_scaled

def overlap(matches, depth_map, max_pixel_dist=2):
    """
    For each (u, v) in matches, find the nearest valid depth pixel within max_pixel_dist.
    Returns:
        matched_uv:        (u,v) from original matches
        matched_lidar_uv:  nearest valid (u,v) from depth map
        matched_indices:   indices into original match list
    """
    valid_mask = np.isfinite(depth_map)
    valid_v, valid_u = np.where(valid_mask)
    valid_uv = np.stack((valid_u, valid_v), axis=-1)
    matched_uv, matched_lidar_uv, matched_indices = [], [], []

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

def get_master_output(model, device, anchor_image, query_image, visualize=False, verbose=True): 
    """Run MASt3R inference and return matches and related data."""
    images = load_images([anchor_image, query_image], size=512, verbose=verbose)
    
    t0 = time.time()
    output = inference([tuple(images)], model, device, batch_size=1, verbose=False)
    t1 = time.time()

    view1, view2 = output['view1'], output['view2']
    pred1, pred2 = output['pred1'], output['pred2']

    desc1 = pred1['desc'].squeeze(0).detach()
    desc2 = pred2['desc'].squeeze(0).detach()

    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8, device=device, dist='dot', block_size=2**13)

    if verbose:
        print(f"MASt3R Inference Time: {t1 - t0:.4f}s")

    ignore_margin = 3
    H0, W0 = view1['true_shape'][0]
    H1, W1 = view2['true_shape'][0]

    # Use H0, W0, H1, W1 as ints in the validity checks
    valid_0 = (matches_im0[:, 0] >= ignore_margin) & (matches_im0[:, 0] < int(W0) - ignore_margin) & \
              (matches_im0[:, 1] >= ignore_margin) & (matches_im0[:, 1] < int(H0) - ignore_margin)
    valid_1 = (matches_im1[:, 0] >= ignore_margin) & (matches_im1[:, 0] < int(W1) - ignore_margin) & \
              (matches_im1[:, 1] >= ignore_margin) & (matches_im1[:, 1] < int(H1) - ignore_margin)

    valid = valid_0 & valid_1
    matches_im0, matches_im1 = matches_im0[valid], matches_im1[valid]

    pts3d_im0 = pred1['pts3d'].squeeze(0).detach().cpu().numpy()
    pts3d_im1 = pred2['pts3d_in_other_view'].squeeze(0).detach().cpu().numpy()
    conf_im0 = pred1['conf'].squeeze(0).detach().cpu().numpy()
    conf_im1 = pred2['conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im0 = pred1['desc_conf'].squeeze(0).detach().cpu().numpy()
    desc_conf_im1 = pred2['desc_conf'].squeeze(0).detach().cpu().numpy()

    if visualize:
        visualize_2d_matches(conf_im0, conf_im1, matches_im0, matches_im1, view1, view2)

    return (matches_im0, matches_im1,
            pts3d_im0, pts3d_im1)

def visualize_2d_matches(conf_im0, conf_im1, matches_im0, matches_im1, view1, view2, n_viz=20):
    """Optional visualization of matches with confidence heatmap."""
    match_idx_to_viz = np.linspace(0, len(matches_im0) - 1, n_viz).astype(int)
    viz_matches_im0 = matches_im0[match_idx_to_viz]
    viz_matches_im1 = matches_im1[match_idx_to_viz]

    mean, std = torch.tensor([0.5]), torch.tensor([0.5])
    imgs = []
    for view in [view1, view2]:
        img = view['img'] * std[:, None, None] + mean[:, None, None]
        imgs.append(img.squeeze(0).permute(1, 2, 0).cpu().numpy())

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img = np.concatenate((imgs[0], imgs[1]), axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.set_title('Top Matches with Confidence')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i], viz_matches_im1[i]
        ax.plot([x0, x1 + W0], [y0, y1], '-+', color=plt.cm.jet(i / n_viz))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin=conf_im0.min(), vmax=conf_im0.max()), cmap='viridis'), cax=cax)
    plt.tight_layout()
    plt.show()

def plot_depth_overlay_on_image(img, scene_map, pixel_uv, cmap='plasma', alpha=1.0, point_size=3, title=None):
    """
    Overlay depth values from a depth map at given pixel coordinates on an image.

    Args:
        img (np.ndarray): Image (H, W, 3) or (H, W).
        depth_map (np.ndarray): Depth map (H, W).
        pixel_uv (np.ndarray): (N, 2) array of (u, v) pixel coordinates.
        cmap (str): Colormap for depth.
        alpha (float): Alpha for scatter.
        point_size (int): Size of scatter points.
        title (str): Plot title.
    """
    u = pixel_uv[:, 0]
    v = pixel_uv[:, 1]
    depth = scene_map[v, u][:,2]
    fig, ax = plt.subplots()
    ax.imshow(img, origin='upper')
    scatter = ax.scatter(u, v, c=depth, cmap=cmap, s=point_size, alpha=alpha)
    plt.colorbar(scatter, ax=ax, label='Depth (m)')
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis('off')
    ax.set_title(title if title else 'Depth Overlay on Image')
    plt.show()

def plot_lidar_mast3r_matches(img, lidar_uv, matched_lidar_uv, inliers_im0, title="LiDAR: Matched vs Unmatched Projections"):
    """
    Visualize matched/unmatched LiDAR projections and MASt3R inlier matches.

    Args:
        img: Image array (background for scatter plot)
        lidar_uv: (N,2) array of all valid LiDAR-projected pixel coordinates
        matched_lidar_uv: (M,2) array of matched LiDAR pixel coordinates
        inliers_im0: (M,2) array of MASt3R inlier keypoints (matched to LiDAR)
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import numpy as np

    matched_lidar_uv_set = set(map(tuple, matched_lidar_uv))
    mask_matched = np.array([tuple(uv) in matched_lidar_uv_set for uv in lidar_uv])
    lidar_matched_uv = lidar_uv[mask_matched]
    lidar_unmatched_uv = lidar_uv[~mask_matched]

    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    if len(lidar_unmatched_uv) > 0:
        plt.scatter(lidar_unmatched_uv[:, 0], lidar_unmatched_uv[:, 1], c='lightgray', s=1, label='Unmatched LiDAR')
    if len(lidar_matched_uv) > 0:
        plt.scatter(lidar_matched_uv[:, 0], lidar_matched_uv[:, 1], c='blue', s=6, label='Matched LiDAR')
    if len(inliers_im0) > 0:
        plt.scatter(inliers_im0[:, 0], inliers_im0[:, 1], c='red', s=6, label='MASt3R Inliers')
        # Draw lines between inlier keypoints and matched LiDAR
        for match_uv, lidar_uv_ in zip(inliers_im0, matched_lidar_uv):
            plt.plot([match_uv[0], lidar_uv_[0]], [match_uv[1], lidar_uv_[1]], color='lime', linewidth=1)
    plt.title(title)
    plt.axis('off')
    plt.legend()
    plt.show()


def get_mast3r_image_shape(original_width, original_height, target_size=512, square_ok=False):
    """
    Compute the resized and center-cropped image dimensions for MASt3R preprocessing.

    Args:
        original_width (int): Original image width.
        original_height (int): Original image height.
        target_size (int): Longest side resized to this (default: 512).
        square_ok (bool): Allow square aspect ratio as output if True (default: False).

    Returns:
        tuple: (processed_height, processed_width)
    """
    if original_width > original_height:
        new_width = target_size
        new_height = int(original_height * (target_size / original_width))
    else:
        new_height = target_size
        new_width = int(original_width * (target_size / original_height))

    cx, cy = new_width // 2, new_height // 2
    halfw = ((2 * cx) // 16) * 8
    halfh = ((2 * cy) // 16) * 8

    if not square_ok and new_width == new_height:
        halfh = int(3 * halfw / 4)

    processed_width = 2 * halfw
    processed_height = 2 * halfh

    return  processed_width,processed_height

def quaternion_rotational_error(q1, q2):
    """
    Compute the rotational error (angular distance) between two quaternions.
    Args:
        q1: Quaternion 1 (array-like, [qx, qy, qz, qw])
        q2: Quaternion 2 (array-like, [qx, qy, qz, qw])
    Returns:
        Angular distance in radians.
    """
    # Normalize the quaternions to ensure they are unit quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Compute the relative quaternion (q1 * q2^-1)
    q_relative = R.from_quat(q1).inv() * R.from_quat(q2)

    # Convert the relative quaternion to an angle
    angle = q_relative.magnitude()  # Returns the angular distance in radians
    return angle