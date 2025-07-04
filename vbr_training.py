from mast3r.datasets.vbr_pairs_dataset import VBRPairsDataset

dataset = VBRPairsDataset(
    root_dir="/home/bjangley/VPR/vbr/spagna_train0_00",
    pairs_txt="/home/bjangley/VPR/mast3r-v2/pairs_test.txt",
    intrinsics="/absolute/path/to/intrinsics.npy",
    poses="/absolute/path/to/poses.npy",
    depth_dir="/absolute/path/to/campus_train0_00_kitti/depthmaps_npy",
    resolution=(384, 384),
    split="train",
    aug_crop=True,
    n_corres=100,
)

# Test loading the first pair
views = dataset[0]

for i, view in enumerate(views):
    print(f"\n--- View {i} ---")
    print(f"Image size:     {view['img'].size}")
    print(f"Depth shape:    {view['depthmap'].shape}")
    print(f"Intrinsics:\n{view['camera_intrinsics']}")
    print(f"Pose:\n{view['camera_pose']}")
