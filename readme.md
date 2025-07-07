
### Mining Pairs Using Inlier Counting via Fundamental Matrix Estimation


#### Sequences in the VBR Spagna Scene with overlap

These are the sequences that have overlap:
| Anchor Sequence | Query Sequences               |
|-----------------|-------------------------------|
| 0               | 3, 10                         |
| 4               | 10, 11, 3, 5, 19, 12          |
| 2               | 20, 21                        |
| 7               | 14                            |
| 13              | 15                            | 


To generate anchor query pairs, inliers are counted using `cv2.fundamentalmatrix()` with matches from MASt3R, a feature matching algorithm. After counting inliers for all pairs from an anchor and query sequence, each query is matched to the top 3 anchors found.

The fundamental matrix is used instead of homography as it 


Example usage:

```
python my_scripts/mining.py \
  --dataset /datasets/vbr_slam/spagna/spagna_train0_kitti \
  --gt /datasets/vbr_slam/spagna/spagna_train0/spagna_train0_gt.txt \
  --calib /datasets/vbr_slam/spagna/spagna_train0/vbr_calib.yaml \
  --sequence_pairs "0:3,10;4:10,11,3,5,19,12;20:2,21;7:14;13:15" \
  --anchor_step 10 \
  --query_step 50 \
  --output results/spagna_matches_inliers_fm.csv \
  --top_n 3 \
  --temp_file results/spagna_processed_pairs.txt
```

- Results include the number of matches and inliers for each pair, saved to the specified output file.
- Sorted pairs (queries matched to their top N anchors) are saved to `{output}_{top_n}.csv`.

---

### Running Localization on Mined Pairs

To localize using the generated anchor-query pairs, use `my_scripts/localization.py`. This script also generates results using scaled MASt3R pointmaps. Scaling is computed using two methods (see `my_utils/scaling.py`):

1. **Uniform Scaling:** L1 norm between centroids of MASt3R and LiDAR 3D points, applied uniformly to all axes.
2. **Per-Axis Scaling:** L1 norm between centroids, computed and applied separately for each axis.

#### Example: Running Localization

```bash
python my_scripts/localization.py \
  --dataset /datasets/vbr_slam/spagna/spagna_train0_kitti \
  --gt /datasets/vbr_slam/spagna/spagna_train0/spagna_train0_gt.txt \
  --calib /datasets/vbr_slam/spagna/spagna_train0/vbr_calib.yaml \
  --pairs_csv results2/spagna_matches_inliers_fm_top3.csv \
  --output_prefix results2/spagna 
```

**Arguments:**
- `--dataset`: Path to the dataset in KITTI format.
- `--gt`: Path to the ground truth file.
- `--calib`: Path to the calibration file.
- `--pairs_csv`: Path to the CSV file with anchor-query pairs.
- `--output_prefix`: Prefix for output files (e.g., `results2/spagna`).

---

### Evaluation Script

Evaluate the localization results using `my_scripts/evaluate.py`:

```bash
python my_scripts/evaluate.py \
  --dataset /datasets/vbr_slam/spagna/spagna_train0_kitti \
  --gt /datasets/vbr_slam/spagna/spagna_train0/spagna_train0_gt.txt \
  --calib /datasets/vbr_slam/spagna/spagna_train0/vbr_calib.yaml \
  --pairs_csv results/spagna_matches_inliers_fm_top3_anchors_per_query_per_anchorseq.csv \
  --output_prefix results_full/spagna \
  --temp_file results_full/spagna_processed_pairs.txt
```


```bash
python my_scripts/evaluate_v2.py \
  --dataset /datasets/vbr_slam/spagna/spagna_train0_kitti \
  --gt /datasets/vbr_slam/spagna/spagna_train0/spagna_train0_gt.txt \
  --calib /datasets/vbr_slam/spagna/spagna_train0/vbr_calib.yaml \
  --pairs_csv mined_step50/spagna_matches_inliers_fm_top3_anchors_per_query_per_anchorseq.csv \
  --output_prefix results_step50/spagna \
  --model_name naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --min_inliers 200 \
  --temp_file results_step50/spagna_processed_pairs.txt
```


**Arguments:**
- `--dataset`: Path to the dataset in KITTI format.
- `--gt`: Path to the ground truth file.
- `--calib`: Path to the calibration file.
- `--pairs_csv`: CSV file with anchor-query pairs.
- `--output_prefix`: Prefix for output files (e.g., `results2/spagna`). Results will be saved as `results2/spagna_mast3r.txt`, `results2/spagna_lidar.txt`, etc.
- `--temp_file`: File to track processed pairs, ensuring previously processed pairs are skipped in subsequent runs.
