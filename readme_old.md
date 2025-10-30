### Dataset Preparation




### Mining Pairs Using Inlier Counting via Fundamental Matrix Estimation

To generate anchor query pairs, inliers are counted using `cv2.fundamentalmatrix()` with matches from MASt3R, a feature matching algorithm. After counting inliers for all pairs from an anchor and query sequence, each query is matched to the top 3 anchors found.

The fundamental matrix is used instead of homography as it 







Example usage:

```
python mining.py --dataset_scene <dataset_scene_name> --anchor_query_json <path_to_anchor_query_json> --anchor_step <anchor_step_size> --query_step <query_step_size> --output <output_csv_file> --top_n <number_of_top_anchors> --device <device_name> --model_name <model_name> --temp_file <temp_file_name>
```

```
python my_scripts/mining.py --dataset_scene spagna --anchor_query_json /home/bjangley/VPR/mast3r-v2/my_vbr_utils/vbr_sequences/spagnav2.json --anchor_step 50 --query_step 50 --output pairs/spagna_v2/spagna_matches_inliers_fm.csv --top_n 3 --temp_file pairs/spagna_v2/processed_pairs.txt

```

```
python my_scripts/mining.py --dataset_scene campus --anchor_query_json /home/bjangley/VPR/mast3r-v2/my_vbr_utils/vbr_sequences/campusv2.json --anchor_step 10 --query_step 20 --output pairs_campus_v2/campus_matches_inliers_fm.csv --top_n 10 --temp_file pairs_campus_v2/processed_pairs.txt
```

```
python my_scripts/mining.py --dataset_scene ciampino --anchor_query_json /home/bjangley/VPR/mast3r-v2/my_vbr_utils/vbr_sequences/ciampino.json --anchor_step 10 --query_step 20 --output results_ciampino/ciampino_matches_inliers_fm.csv --top_n 3 --temp_file results_ciampino/processed_pairs.txt
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

/home/bjangley/VPR/mast3r-v2/pairs_campus_v2/campus_matches_inliers_fm_top3_anchors_per_query.csv
```bash
python my_scripts/evaluate_v2.py \
  --dataset_scene spagna \
  --pairs_csv pairs/spagna_v2/spagna_matches_inliers_fm_top10_anchors_per_query.csv \
  --output_prefix results_localisation/spagna_pairsv2_top10/spagna \
  --model_name naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --min_inliers 200 \
  --temp_file results_localisation/spagna_pairsv2_top10/spagna_processed_pairs.txt
```

```
python my_scripts/evaluate_v2.py \
  --dataset_scene campus \
  --pairs_csv pairs_campus_v2/campus_matches_inliers_fm_top10_anchors_per_query.csv \
  --output_prefix results_campusv3_pairsv2_top10/campus \
  --model_name naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --min_inliers 200 \
  --temp_file results_campusv3_pairsv2_top10/campus_processed_pairs.txt
```

python my_scripts/evaluate_v2.py \
  --dataset_scene ciampino \
  --pairs_csv pairs/pairs_ciampinov1/ciampino_matches_inliers_fm_top3_anchors_per_query.csv \
  --output_prefix results_localisation/ciampino_v1 \
  --model_name naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --min_inliers 200 \
  --temp_file  results_localisation/ciampino_processed_pairs.txt


/home/bjangley/VPR/mast3r-v2/pairs_campus_v2/campus_matches_inliers_fm_top3_anchors_per_query.csv

```
python my_scripts/evaluate.py \
  --dataset /datasets/vbr_slam/spagna/spagna_train0_kitti \
  --gt /datasets/vbr_slam/spagna/spagna_train0/spagna_train0_gt.txt \
  --calib /datasets/vbr_slam/spagna/spagna_train0/vbr_calib.yaml \
  --pairs_csv results_step50_v2/spagna_matches_inliers_fm_top3_anchors_per_query.csv\
  --output_prefix results_step50v4/spagna \
  --model_name naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --min_inliers 200 \
  --temp_file results_step50v4/spagna_processed_pairs.txt
```

**Arguments:**
- `--dataset`: Path to the dataset in KITTI format.
- `--gt`: Path to the ground truth file.
- `--calib`: Path to the calibration file.
- `--pairs_csv`: CSV file with anchor-query pairs.
- `--output_prefix`: Prefix for output files (e.g., `results2/spagna`). Results will be saved as `results2/spagna_mast3r.txt`, `results2/spagna_lidar.txt`, etc.
- `--temp_file`: File to track processed pairs, ensuring previously processed pairs are skipped in subsequent runs.



python my_scripts/evaluate.py \
  --dataset_scene campus_train0 \
  --pairs_path pairs_finetuning/campus_train0/test_pairs.txt \
  --output_prefix results_finetuning/campus_train0/campus_train_0 \
  --model_path /home/bjangley/VPR/mast3r-v2/checkpoints/campus/checkpoint-best.pth \
  --min_inliers 200 \
  --temp_file results_finetuning/campus_train0/campus_train_0_processed_pairs.txt