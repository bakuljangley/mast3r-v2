
# Dataset Preparation

### The Dataset Format
This repository provides a flexible Python interface for loading and synchronizing the VBR SLAM dataset, as released by the [RVP Group from Sapienza University of Rome](https://rvp-group.net). To download the dataset, follow the [official instructions](https://github.com/rvp-group/vbr-devkit?tab=readme-ov-file) (relevant scenes campus, spagna, ciampino and colosseo) and convert them to the KITTI Format.

The final directory structure should be:
```
/datasets/vbr_slam/
├── campus/
│   ├── campus_train0_kitti/
│   │   ├── camera_left/ (PNG format, numbered sequentially)
│   │   │   ├── data/           # Left camera images (*.png)
│   │   │   └── timestamps.txt  # Left camera timestamps 
|   |   |       (Timestamps: Text files with one timestamp per line in ISO format)
│   │   ├── camera_right/
│   │   │   ├── data/           # Right camera images (*.png)
│   │   │   └── timestamps.txt  # Right camera timestamps
│   │   └── ouster_points/ (Binary files containing structured point cloud data)
│   │       ├── data/           # LiDAR point clouds (*.bin)
│   │       ├── timestamps.txt  # LiDAR timestamps
│   │       └── .metadata.json  # Point cloud metadata
│   └── campus_train0/
│       └── campus_train0_gt.txt # Ground truth poses (timestamp tx ty tz qx qy qz qw)
|   |___campus_train1_kitti/
|   |___campus_train1/  
├── spagna/
├── ciampino/
└── colosseo/
```


### Interpolation Details
The dataset performs temporal interpolation to synchronize sensors with different sampling rates. All data is aligned to the camera timestamps as the reference

Ground truth poses are interpolated using:
1. Translation: Linear interpolation between nearest pose timestamps
1. Rotation: Spherical Linear Interpolation (SLERP) for quaternions

#### LiDAR Interpolation
LiDAR point clouds are interpolated by:
1. Finding the two nearest LiDAR scans to the image timestamp
1. Performing linear interpolation on XYZ coordinates of corresponding points
Parameters:
    - max_time_diff: Maximum allowed time difference for interpolation (default: 0.1 seconds)
    - Fallback behavior: Returns None or default values when interpolation fails
-- MOST RECENT EDIT: I turned off the interpolation when I realised either they've already accounted for it + the time differences are minute. Using interpolation causes weird points to appear.

## Ground Truth GPS Location Generation

The VBR dataset only provides a local trajectory (local frame is lidar frame at $t=0$) with timestamps and the pose translations and quaternion.

For certain manually selected images in the dataset, I selected markers (such as zebra crossings, corners, etc) visible on the image and satellite maps. These locations were used to localise the images on the global map and generate GPS locations for the rest of the images in the sequence.


### Steps:

1. **Input:**
   * A set of 2D pixel locations in the image.
   * Corresponding GPS coordinates (latitude, longitude) for those 2D points.
   * Camera intrinsics (`K`) and distortion coefficients (`dist`).

2. **Convert GPS to UTM:**  Each GPS coordinate is converted to UTM (Easting - x, Northing - y) using `pyproj`.

3. **Form 3D Object Points:** The converted UTM points are treated as 3D coordinates on a flat plane (Z = 0).

4. **PnP:** To find the transformation from UTM to camera frame (`T_cam_utm`)

5. **Estimate Camera Position in UTM Frame:** by converting back to latitude and longitude.
     $$\text{Camera\_pos}_{\text{UTM}} = -R^T \cdot t$$

#### **Coordinate Frame Alignment:**
From the dataset, we obtain the ground truth camera pose `T_local_cam` for the same image (camera in local frame).

$T_{\text{UTM} \leftarrow \text{Local}} = T_{\text{UTM} \leftarrow \text{Cam}} \cdot T_{\text{Cam} \leftarrow \text{Local}}$

However, directly applying the transformation from each individually localized image can result in inconsistent orientations and misaligned axes between the local and global (UTM) frames. To resolve this, a similarity transform is computed, which finds the optimal 2D rotation, scale, and translation that best aligns the 5 localised points (with local poses defined at the origin of the sequence) with the global UTM trajectory. This ensures that the axes and scale of the local frame are consistently matched to those of the global frame across all images.

After localizing 4-5 images using steps 1-5, a rotation may exist between the local poses and UTM localizations. The similarity transform aligns two sets of points by estimating a scale, rotation, and translation that best map one to the other. First, both point sets are centered by subtracting their means. The algorithm then computes the optimal rotation using Singular Value Decomposition (SVD) on their covariance matrix. The scale factor is computed as the ratio of their centered norms, and the translation aligns their centroids after scaling and rotation. This results in a transformation

$\mathbf{x}_{\text{global}} = s \cdot R \cdot \mathbf{x}_{\text{local}} + \mathbf{t}$

that maps local coordinates into the UTM frame, aligning the axes and scales of both coordinate systems, where:

- **$\mathbf{x}_{\text{local}}$**: point in the local coordinate frame  
- **$\mathbf{x}_{\text{global}}$**: corresponding point in the global (UTM) frame





