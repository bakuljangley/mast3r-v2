# Master's Thesis in Robotics, TU Delft

[**On the Generalization of Metric Relative Pose Estimation Models to Unseen Environments**](https://repository.tudelft.nl/record/uuid:f8b0899b-a921-4d6e-8593-9942a8388301) 

- Student Name: **Bakul Jangley**
- Supervisors: **Prof. Julian Kooij, Mubariz Zaffar** (Intelligent Vehicles Group)
- [Link to thesis](https://repository.tudelft.nl/record/uuid:f8b0899b-a921-4d6e-8593-9942a8388301) 

Crowd-sourced imagery is increasingly important for urban mapping and visual localization. However, its reliability is limited by GPS inaccuracies and heterogeneous capture condi- tions, including device variability, viewpoint differences, illumination changes, and temporal shifts. In these settings, achieving metric-scale pose estimation remains a central challenge. Deep Learning-based pose estimation models address this problem by learning to estimate the 6-DoF pose using geometric cues between image views and metric supervision during training on large datasets. This encourages spatial consistency and supports generalization across diverse conditions. Recent learning--based architectures, often based on vision transformer encoders, approach the task through unified multi-task frameworks that jointly predict metric depthmaps and 2D–2D correspondences, with relative pose estimated downstream. This thesis evaluates whether such frameworks predict accurate metric depthmaps under domain shifts. Experiments show that, even with scale correction through data-driven fine-tuning with metric supervision, depth predictions from multi-task relative pose estimation models fail to generalize reliably to out-of-domain environments. In contrast, monocular models, trained on significantly larger and more varied datasets, demonstrate strong zero-shot reliability for metric depth prediction. A hybrid pipeline is proposed that combines the geometric consistency of relative pose models with the stable metric cues of monocular models, enabling robust pose estimation in crowd-sourced outdoor environments.


## Overview
This repository studies generalization of metric relative-pose estimation models ([MASt3R](https://github.com/naver/mast3r)) under domain shift and compares them with large-scale monocular metric depth models. Key contributions:
- Evaluation of MASt3R-style multi-task models for metric depth and relative pose in out-of-domain outdoor scenes.
- Empirical finding: multi-task relative-pose models often fail to generalize metric depth; monocular models trained on large diverse datasets show stronger zero-shot metric depth performance.
- A hybrid pipeline combining monocular depth for stable metric cues with MASt3R geometric matching for robust pose estimation.

Contents:
- Setup and environment instructions
- Dataset preparation (VBR Rome, Mapillary)
- Fine-tuning MASt3R
- Evaluation scripts and result compilation
- Output formats and result interpretation
- Citation and references

## Quick Start: Environment Set Up

**DISCLAIMER**: different models require separate environments. Default environment: `mast3r`.
For the subsequent sections, most of my code uses bash scripts that will require minor changes in path variables or settings (such as scene, hyper-params, etc.) need to be changed, then make scripts executable and run:

```
chmod+x <filename.sh>
./<filename.sh>
``` 

### MASt3R:
- Clone this repository recursively: 
    ```
    git clone --recursive https://github.com/bakuljangley/mast3r-v2
    ``` 
- Follow the MASt3R environment set up as provided by the [official implementation](https://github.com/naver/mast3r). 
- Download the `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` checkpoint using: 
    ```
    mkdir -p checkpoints/
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
    ```


### Monocular Depth Estimation Models:
As explained before, use separate environments for each model. Examples using conda:
1. [ZoeDepth](https://github.com/isl-org/ZoeDepth):
    ```
    cd mast3r-v2/monocular_depth_models/zoedepth
    conda env create -n zoe --file environment.yml
    conda activate zoe  
    ```
2. [DepthPro](https://github.com/apple/ml-depth-pro):
    ```
    cd mast3r-v2/monocular_depth_models/depthpro
    conda create -n depth-pro -y python=3.9
    conda activate depth-pro
    pip install -e .
    ```
3. [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2): 
    ```
    cd mast3r-v2/monocular_depth_models/depthanythingv2
    conda create --name depthanything python=3.9
    conda activate depthanything
    pip install -r requirements.txt
    ```




## Dataset Preparation


### 1. VBR, Rome 
Download the [VBR Rome](https://github.com/rvp-group/vbr-devkit) (2025) dataset. This project only uses the following trajectories, convert them into KITTI format after downloading. The VBR dataset is built from multiple trajectories in each scene, my code treats each trajectory individually and then compiles per scene results afterwards (grouping trajectories, if necessary). The table below reports how scenes are constructed and the different sampling and top_n values used for each trajectory.

| Scene      | Trajectory                               | Sub-sampling (Query/Anchor) | Top n Anchors | Total Pairs | Valid Pairs (>200 Inliers) |
|------------|------------------------------------------|--------------------|----------------|--------------|-----------------------------|
| Spagna     | `spagna_train0`                          | 50 / 50            | 10             | 2580         | 2485                        |
| Campus     | `campus_train0`                          | 20 / 10            | 10             | 2060         | 1929                        |
|            | `campus_train1`                          | 20 / 10            | 10             | 2060         | 1929                        |
| Ciampino 1 | `ciampino_train0`                        | 20 / 10            | 5              | 2060         | 2030                        |
| Ciampino 2 | `ciampino_train1`                        | 20 / 10            | 7              | 2212         | 1992                        |


1. **Downloading and Pre-Processing the VBR Rome Dataset**

    Install helper:
    ```
    pip install vbr-devkit
    ```

    Download + convert to KITTI-like:
    ```
    vbr download <sequence_name> <save_directory>
    vbr convert kitti <input_bag> <output_directory>
    ```
2. **Anchor-Query Pair Mining** (Optional, slow) : `mining.sh`
    - Required variables (example values shown):
        - `SCENE="ciampino_train1"` — trajectory to evaluate
        - `DATASET_ROOT="/datasets/vbr_slam"` — root directory for the VBR/KITTI dataset
        - `SEQUENCE_PATH="my_vbr_utils/vbr_sequences/ciampino_train1.json"` — path to the JSON specifying image indices / sequence layout

    The indices of the anchor-query images compared are provided in `my_vbr_utils/vbr_sequences/<trajectory>.json`. This will create an output directory `pairs_mining/<scene>/` and store all anchor-query pairs and top-n pairs as `.csv` files.   
3. **Train/Val/Test1/Test2 Splits + Additional Supervision Data** (Required for fine-tuning) : `prepare_mast3r_dataset.sh`
    - Required variables (example values shown):
        - `DATASET_ROOT="/datasets/vbr_slam"` — root directory for the VBR/KITTI dataset
        - `DATASET_SCENE="ciampino_train0"` — scene/trajectory used to build splits
        - `TOP_N=5` — number of top anchors retained per query when constructing pairs
        - `PAIRS_ROOT="/pairs_mining/"` — root path to the mined pairs produced by `mining.sh`
        - `OUTPUT_DIR="/pairs_finetuning/"` — output root for split files (script creates scene-wise subfolders with `.txt` files for each split)
        - `OUTPUT_ROOT="/vbr"` — path where generated depthmaps and poses will be saved

    This saves VBR depthmaps as `.npy` files and re-generates pose labels (for each image's time-stamp) under the `OUTPUT_DIR`.
4. **VBR Global Alignment** : (required for Mapillary Experiments)
    To generate GPS positions for VBR trajectories use `generate_global_trajectories.sh` (change the `DATASET_ROOT` ,`UTILS_ROOT` and `SAVE_ROOT`, if required). This uses the manually selected pixels and cross-view (satellite) correspondences provided in `my_vbr_utils/GPSalignment`. 
    
    For trouble shooting or additional details about the dataset setup code, refer to [this](https://github.com/bakuljangley/mast3r-v2/blob/main/my_vbr_utils/vbrDatasetreadme.md). Additionally, refer to the provided [output formats](#output-generation).

### 2. Mapillary Images
1. **Download**:
    If you want to download from scratch (optional), this repository already provides the Mapillary images used.
    ```
    cd mapillary
    python vbr_downloads.py #Using the official API
    ```
    The code uses my personal authentication token from the Mapillary website, if the download does not work or stalls, the best solution is to generate your own and replace it in `mapillary/utils.py`. 
2. **Mining Anchor-Query Pairs**: Run the `mine_mapillary.sh` bash script after replacing `MAPILLARY_ROOT` and `OUTPUT_ROOT`. 




## Fine-tuning MASt3R-DPT

**Prerequisites**:
- Precomputed depthmaps and poses (from [dataset preparation](#1-vbr-rome))
- MASt3R checkpoint

The `run_finetuning.sh` script runs finetuning and can be configured for different training and validation sets, the loss function can also be changed. The code is similar to the original implementation of [`train.py`](https://github.com/naver/mast3r). 

- **Key parameters to edit in the script:** (example values shown)
    - `ROOT="/datasets/vbr_slam/"`— root directory for the VBR/KITTI dataset
    - `PAIRS_PATH="mast3r-v2/pairs_finetuning/"` — root directory to the pairs generated for finetuning
    - `VBR_LABELS="/vbr"` — path to prepared depthmap and pose folder
    - Dataset Arguments
        - Scene Names: 
            - `SCENE_TRAIN1`, `SCENE_TRAIN2` (multiples defined when trajectories need to be concatenated)
            - `SCENE_VAL`
        - Dataset Construction: 
            - `TRAIN_DATASET`
            - `TEST_DATASET`
    - `TRAIN_CRITERION` — Loss Function 
        - Regression Loss: `"Regr3D(L21, norm_mode='?avg_dis', gt_scale=True)"` 
        - Confidence Weighed Regression Loss: `"ConfLoss(Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0),alpha=0.2)"`
    - `PRETRAINED="/path/to/model_checkpoint.pth` — Pre-trained Model Path 
    - `OUTPUT_DIR_CHECKPOINTS="<checkpoints_dir>/<training_details>` — Output Directory
    - `LR`, `MIN_LR` — Learning Rates
- **Exact specifications to used to generate training datasets:**
    - **MASt3R-DPT (Campus &rarr; Ciampino2)** *trained with regression loss*.
        ```
        SCENE_TRAIN1="campus_train0"
        SCENE_TRAIN2="campus_train1"
        SCENE_VAL="ciampino_train1"
        TRAIN_DATASET="VBRPairsDataset(root_dir='$ROOT',scene='$SCENE_TRAIN1', split='train', pairs_dir='$PAIRS_PATH', depth_dir='$DEPTH_DIR' , pose_dir='$POSE_DIR', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=False)+VBRPairsDataset(root_dir='$ROOT',scene='$SCENE_TRAIN2', split='train', pairs_dir='$PAIRS_PATH', depth_dir='$DEPTH_DIR' , pose_dir='$POSE_DIR', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=False)"
        TRAIN_CRITERION="Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)"
        ```
    - **MASt3R-DPT (Campus &rarr; Ciampino2)** *trained with regression loss*.
        ```
        SCENE_TRAIN1="ciampino_train0"
        SCENE_VAL="ciampino_train1"
        TRAIN_DATASET="VBRPairsDataset(root_dir='$ROOT',scene='$SCENE_TRAIN1', split='train', pairs_dir='$PAIRS_PATH', depth_dir='$DEPTH_DIR' , pose_dir='$POSE_DIR', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=False)"
        TRAIN_CRITERION="Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)"
        ```
- The training curves can be inspected using `training_curves.ipynb` by changing the `base_path` to the `OUTPUT_DIR_CHECKPOINTS` for your fine-tuned model.


## Experiments on VBR

This section outlines how to re-produce the results of Relative Pose Estimation on the VBR Rome dataset. For all experiments, I use bash scripts to execute python files. To re-produce my results


### 1. Using MASt3R as H and G 

**Prerequisites**: fine-tuning data pre-processing (specifically the dataset splits `.txt` files - which are also provided by default)

The `run_evaluate.sh` script will run PnP using a specified MASt3R checkpoint as both G and H. 

- **Configuration** (example values shown — edit these in `run_evaluate.sh` before running):
    - `SCENES="ciampino_train0"` — trajectory(s) to evaluate (comma/space-separated allowed)
    - `DATASET_ROOT="/datasets/vbr_slam"` — root directory for the VBR/KITTI dataset
    - `PAIRS_PATH="pairs_finetuning/"` — root directory containing fine-tuning pairs
    - `SPLIT="all"` — dataset split to evaluate (e.g., `train`, `val`, `test1`, `test2`, `all`)
    - `OUTPUT_ROOT="results_localization/original/"` — base output directory; results are saved under `OUTPUT_ROOT/<scene>/`
    - `CHECKPOINT="/path/to/model_checkpoint.pth"` — MASt3R checkpoint to use (use fine‑tuned checkpoint to compile finetuned results)
- Important notes:
    - The evaluation logic and available scaling / H function choices (including Oracle, scaled versions of MASt3R) are defined in `my_scripts/evaluate_v5.py`. Review `METHOD_CONFIG` there to change behaviour.
    - To apply precomputed train-set scaling factors (saved in `my_vbr_utils/train_scales.json`), run `apply_train_scales.sh` after or before evaluation as appropriate.
    - **CAUTION**: set `conf_percentile` correctly in the final command (use `0` for no confidence filtering).
    - **Batching recommendation**: run separate scenes in separate terminals — each scene can be slow. *Both `run_evaluate.sh` and `apply_train_scales.sh` will need to be re-run for each of the 5 trajectories, you could run all scenes in one go but it's not recommended as each scene takes a while, it's better to run multiple processes for different `scenes` in separate terminals.*


### 2. Using Monocular Depth Estimation Models as H and MASt3R as G

1. **Generating Depthmaps**: After setting up environments for the metric monocular depth estimation models as described [earlier](#monocular-depth-estimation-models), pre-compute depthmaps for the final anchor-query pairs. *For each bash script, ensure that the location for `PAIRS_BASE_PATH` should be at the root directory where the the fine-tuning splits are stored, and change output directory `OUTPUT_BASE_DIR` as required.*
    - ZoeDepth:
        ```
        cd monocular_depth_models/zoedepth
        conda activate zoe
        chmod+x vbr_depthmaps.sh #change pairs_finetuning, output directories in the bash file
        ./vbr_depthmaps.sh
        ```
    - DepthPro
        ```
        cd monocular_depth_models/depthpro
        conda activate depthpro
        chmod+x generate_vbr_depthmaps.sh #change pairs_finetuning, output directories in the bash file
        ./generate_vbr_depthmaps.sh
        ```
    - Depth Anything V2:
        ```
        cd monocular_depth_models/depthanythingv2/metric_depth
        conda activate depthpro
        chmod+x vbr_depthmaps.sh #change pairs_finetuning, output directories in the bash file
        ./vbr_depthmaps.sh
        ```
2. **Running PnP**: Change the location of the generated depthmaps (from step1) in the `my_scripts/evaluate_anydepth.py` and use `run_evaluate_anydepth.sh`. Edit the output and checkpoint paths. 

    *This will automatically run the script for all trajectories and does not need to be repeated.*

## Compiling Results



## Experiments on Mapillary Images 
**Prerequisites**: mapillary images downloaded

1. **Run PnP**: Use the `localize_mapillary.sh` script to localize Mapillary images, change the `MAPILLARY_ROOT`, `PAIRS_ROOT` and `OUTPUT_ROOT` to suitable locations.
2. **Inspect Results:** 


## Bibliography

This work uses original code provided by the official implementation of `Grounding Image Matching in 3D with MASt3R`  
[[Project page](https://europe.naverlabs.com/blog/mast3r-matching-and-stereo-3d-reconstruction/)], [[MASt3R arxiv](https://arxiv.org/abs/2406.09756)], [[DUSt3R arxiv](https://arxiv.org/abs/2312.14132)]. The code can be found [here](https://github.com/naver/mast3r).
  


```bibtex
@misc{mast3r_eccv24,
      title={Grounding Image Matching in 3D with MASt3R}, 
      author={Vincent Leroy and Yohann Cabon and Jerome Revaud},
      booktitle = {ECCV},
      year = {2024}
}

@misc{mast3r_arxiv24,
      title={Grounding Image Matching in 3D with MASt3R}, 
      author={Vincent Leroy and Yohann Cabon and Jerome Revaud},
      year={2024},
      eprint={2406.09756},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{dust3r_cvpr24,
      title={DUSt3R: Geometric 3D Vision Made Easy}, 
      author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
      booktitle = {CVPR},
      year = {2024}
}

@inproceedings{
    duisterhof2025mastrsfm,
    title={{MAS}t3R-SfM: a Fully-Integrated Solution for Unconstrained Structure-from-Motion},
    author={Bardienus Pieter Duisterhof and Lojze Zust and Philippe Weinzaepfel and Vincent Leroy and Yohann Cabon and Jerome Revaud},
    booktitle={International Conference on 3D Vision 2025},
    year={2025},
    url={https://openreview.net/forum?id=5uw1GRBFoT}
} 
```


# Output Generation
This section broadly explains the expected behaviour upon pairs generation and directory structure to expect the results in. The code that compiles the results is confusing to read, these instructions are to ensure that the output directory structure is consistent.

## PnP Output

1. **Using LiDAR, MASt3R, and MASt3R (Scaled with LiDAR at Test Time)**  

    All PnP computations follow the logic in `my_scripts/evaluate_v5.py`, where the `METHOD_CONFIG` defines the different **H functions**.  
    Each method (Oracle, Scaling, MASt3R) generates separate `.csv` result files.

    - Run this using the `run_evaluate.sh` script (as described [earlier](#using-mast3r-as-h-and-g)). Specify the output directory as:  
        ```
        OUTPUT_ROOT="<results>/<model_type>/"
        ```
    - Use distinct `model_name` values to:
        - Apply **train-set scales**
        - Use **fine-tuned models**
        - Change **confidence thresholds**
        - Or run with the **original pre-trained checkpoint**
    - *Example:* If `OUTPUT_ROOT="results_localization/original/"`, results will be stored as:
        ```
        results_localization/
        ├── original/
        │   ├── campus_train0/
        │   ├── campus_train1/
        │   ├── ciampino_train0/
        │   ├── ciampino_train1/
        │   ├── spagna_train0/
        │   ├── spagna_train0/
        │   │   ├── lidar.csv
        │   │   ├── mast3r.csv
        │   │   ├── mast3r_scaled_icp.csv
        │   │   ├── mast3r_scaled_v2.csv
        │   │   ├── mast3r_scaled_v3.csv
        │   │   └── mast3r_scaled_v4.csv
        │   └── temp_spagna_train0_processed_pairs.txt
        ```
2. **Monocular Depth Prediction Models as H** : Use the same `OUTPUT_ROOT` as the pre-trained MASt3R checkpoint to append monocular model results to the same directory.

    ```
    results_localization/
    ├── original/
    │   ├── campus_train0/
    │   ├── campus_train1/
    │   ├── ciampino_train0/
    │   ├── ciampino_train1/
    │   ├── spagna_train0/
    │   ├── spagna_train0/
    │   │   ├── anydepth80.csv ## 
    │   │   ├── depthpro.csv. ##
    │   │   ├── lidar.csv
    │   │   ├── mast3r.csv
    │   │   ├── mast3r_scaled_icp.csv
    │   │   ├── mast3r_scaled_v2.csv
    │   │   ├── mast3r_scaled_v3.csv
    │   │   └── mast3r_scaled_v4.csv
    │   │   └── zoedepth.csv ##
    │   └── temp_spagna_train0_processed_pairs.txt
    ```

## Pairs Generation 
1. **Mining Anchor-Query Pairs in VBR**: Example output using `mining.sh` will look like this
    ```
    pairs_mining/
    ├── campus_train0
    │   ├── matches_inliers_fm.csv #all compared pairs
    │   ├── matches_inliers_fm_top10_anchors_per_query.csv #top_n pairs
    │   └── processed_pairs.txt
    ├── campus_train1
    ├── ciampino_train0
    ├── ciampino_train1
    └── spagna_train0
    ```
2. **Train/Val/Test1/Test2**: Pairs after pre-processing and shuffling would look like this
    ```
    pairs_finetuning/
    ├── campus_train0
    │   ├── all_pairs.txt
    │   ├── test1_pairs.txt
    │   ├── test2_pairs.txt
    │   ├── train_pairs.txt
    │   └── val_pairs.txt
    ├── campus_train1
    ├── ciampino_train0
    ├── ciampino_train1
    └── spagna_train0
    ```
    Additionally, the output for the depthmaps and pose would be organised like this:
    ```
    vbr/
    ├── poses/
    │   ├── campus_train0.txt #these labels are interpolated to the image timestamps to avoid computation at run-time 
    │   ├── campus_train1.txt
    │   ├── ciampino_train0.txt
    │   ├── ciampino_train1.txt
    │   └── spagna_train0.txt
    └── depths/ 
        ├── campus_train0/ #folders containing .npy depthmaps
        ├── campus_train1/
        ├── ciampino_train0/
        ├── ciampino_train1/
        └── spagna_train0/
    ```
    Depthmaps generated using monocular models are also stored in the same format.
## Extra : Mapillary

tree results_localization/ -L 2 --dirsfirst