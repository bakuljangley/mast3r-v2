# Master's Thesis in Robotics, TU Delft

[**On the Generalization of Metric Relative Pose Estimation Models to Unseen Environments**](https://repository.tudelft.nl/record/uuid:f8b0899b-a921-4d6e-8593-9942a8388301) 

- Student Name: **Bakul Jangley**
- Supervisors: **Prof. Julian Kooij, Mubariz Zaffar** (Intelligent Vehicles Group)

Crowd-sourced imagery is increasingly important for urban mapping and visual localization. However, its reliability is limited by GPS inaccuracies and heterogeneous capture condi- tions, including device variability, viewpoint differences, illumi- nation changes, and temporal shifts. In these settings, achieving metric-scale pose estimation remains a central challenge. Deep Learning-based pose estimation models address this problem by learning to estimate the 6-DoF pose using geometric cues between image views and metric supervision during training on large datasets. This encourages spatial consistency and supports generalization across diverse conditions. Recent learning--based architectures, often based on vision transformer encoders, approach the task through unified multi-task frameworks that jointly predict metric depthmaps and 2D–2D correspondences, with relative pose estimated downstream. This thesis evaluates whether such frameworks predict accurate metric depthmaps under domain shifts. Experiments show that, even with scale correction through data-driven fine-tuning with metric supervision, depth predictions from multi-task relative pose estimation models fail to generalize reliably to out-of-domain environments. In contrast, monocular models, trained on significantly larger and more varied datasets, demonstrate strong zero-shot reliability for metric depth prediction. A hybrid pipeline is proposed that combines the geometric consistency of relative pose models with the stable metric cues of monocular models, enabling robust pose estimation in crowd-sourced outdoor environments.

# Environment Set Up

**DISCLAIMER**: Since the code uses many different models, I recommend setting up different environments to use each model i.e. MASt3R, DepthAnythingV2, ZoeDepth and DepthPro. 

### MASt3R:
- Clone this repository recursively:  `git clone --recursive https://github.com/bakuljangley/mast3r-v2`. 
- Follow the MASt3R environment set up as provided by the [official implementation](https://github.com/naver/mast3r). 
    - Download the `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` checkpoint using: 
        ```
        mkdir -p checkpoints/
        wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
        ```
    - Additionally,

### Monocular Depth Estimation Models:
As explained before, use separate environments for each model. 
1. ZoeDepth :
2. DepthPro:
    ```
    cd mast3r-v2/monocular_depth_models
    conda create -n depth-pro -y python=3.9
    conda activate depth-pro
    pip install -e .
    ```
3. Depth Anything V2:
    ``
    cd mast3r-v2/monocular_depth_models
    conda create --name depthanything python=3.9
    conda activate depthanything
    pip install -r requirements.txt
    ``


# Dataset Preparation

### Downloading and Pre-Processing the VBR Rome Dataset

```
pip install vbr-devkit
vbr --install-completion #optional
vbr download <sequence_name> <save_directory>
vbr convert kitti <input_directory/input_bag> <output_directory>
```

Download the [VBR Rome](https://github.com/rvp-group/vbr-devkit) (2025) dataset. This project only uses the following trajectories, convert them into KITTI format after downloading. The VBR dataset is built from multiple trajectories in each scene, my code treats each trajectory individually and then compiles per scene results afterwards (grouping trajectories, if necessary).

### Anchor-Query Pair Mining (Optional)

| Scene      | Trajectory                               | Sub-sampling (Query/Anchor) | Top n Anchors | Total Pairs | Valid Pairs (>200 Inliers) |
|------------|------------------------------------------|--------------------|----------------|--------------|-----------------------------|
| Spagna     | `spagna_train0`                          | 50 / 50            | 10             | 2580         | 2485                        |
| Campus     | `campus_train0`                          | 20 / 10            | 10             | 2060         | 1929                        |
|            | `campus_train1`                          | 20 / 10            | 10             | 2060         | 1929                        |
| Ciampino 1 | `ciampino_train0`                        | 20 / 10            | 5              | 2060         | 2030                        |
| Ciampino 2 | `ciampino_train1`                        | 20 / 10            | 7              | 2212         | 1992                        |


```
chmod+x mining.sh
./mining.sh #the example script mines pairs for the ciampino1 scene
```

The `mining.sh` script can be used to generate mined pairs (also provided in `pairs_mining`). Running the mining script can take a long time over the entire dataset, it's recommended to use the pairs already generated. The indices of the anchor-query indices compared are provided in `my_vbr_utils/vbr_sequences/<trajectory>.json`. 

### Train/Val/Test1/Test2 Splits + Additional Supervision Data (Required)

To generate dataset splits (also provided in `pairs_finetuning`) and generating supervision labels for finetuning (save VBR depthmaps as `.npy` files and re-generating pose labels):

```
chmod+x prepare_mast3r_dataset.sh
./prepare_mast3r_dataset.sh 
```
*change the save directories and top_n parameter. This process is quite fast and using the mined pairs from `pairs_mining` would allow the code to work. 

### VBR Global Alignment 

To generate GPS positions for VBR trajectories: 
```
chmod+x generate_global_trajectories.sh
./generate_global_trajectories.sh
```
This uses the manually selected pixels and cross-view (satellite) correspondences provided in `my_vbr_utils/GPSalignment`. 

# Fine-tuning MASt3R-DPT

To finetune MASt3R, you must have a downloaded checkpoint stored locally and followed the [pre-processing steps](#trainvaltest1test2-splits--additional-supervision-data-required) i.e. generated ground truth depthmaps and pose labels, and also split the dataset into train/val/test1/test2.

The `run_finetuning.sh` script runs finetuning and can be configured for different training and validation sets, the loss function can also be changed. Exact specifications to use:
- MASt3R-DPT(Campus-->Ciampino2) *trained with regression loss*
    ```
    SCENE_TRAIN1="campus_train0"
    SCENE_TRAIN2="campus_train1"
    SCENE_VAL="ciampino_train1"
    TRAIN_DATASET="VBRPairsDataset(root_dir='$ROOT',scene='$SCENE_TRAIN1', split='train', pairs_dir='$PAIRS_PATH', depth_dir='$DEPTH_DIR' , pose_dir='$POSE_DIR', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=False)+VBRPairsDataset(root_dir='$ROOT',scene='$SCENE_TRAIN2', split='train', pairs_dir='$PAIRS_PATH', depth_dir='$DEPTH_DIR' , pose_dir='$POSE_DIR', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=False)"
    TRAIN_CRITERION="Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)"
    ```
- MASt3R-DPT(Campus-->Ciampino2) *trained with regression loss*
    ```
    SCENE_TRAIN1="ciampino_train0"
    SCENE_VAL="ciampino_train1"
    TRAIN_DATASET="VBRPairsDataset(root_dir='$ROOT',scene='$SCENE_TRAIN1', split='train', pairs_dir='$PAIRS_PATH', depth_dir='$DEPTH_DIR' , pose_dir='$POSE_DIR', resolution=[(512, 384), (512, 336), (512, 288), (512, 256), (512, 160)], aug_crop=False)"
    TRAIN_CRITERION="Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0)"
    ```
- To train a model with confidence weighed regression loss, switch the loss function `TRAIN_CRITERION="ConfLoss(Regr3D(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0),alpha=0.2)"`
- The training curves can be inspected using 

After ensuring these specifications, run the `run_finetuning.sh` script from CLI. 


# Experiments

This section explains how to reproduce the results presented. There are mainly 3 different variants of the depth prediction network used in this work:
1. **MASt3R-DPT** 
2. **MASt3R-DPT (scaled)**   
2. **MASt3R-DPT (finetuned)**
3. **Monocular Depth Prediction Networks**
- Oracle (LiDAR): ground truth depth map 

## Relative Pose Estimation Using MASt3R as G and H 

```
chmod+x run_evaluate.sh 
./run_evaluate.sh
```
The `run_evaluate.sh` script will run PnP using a specified MASt3R checkpoint as both G and H. It also computes results using LiDAR and scaling methods specified (and configurable) in `my_scripts/evaluate_v5.py`. The output will be saved in the `OUTPUT_ROOT` under a folder named after the `scenes` used.

1. `CHECKPOINT`: Specify model checkpoint (to compile results for fine-tuned model run using fine-tuned checkpoint) 
2. `scenes`: `campus_train0`, `campus_train1`, `ciampino_train0`, `ciampino_train0`, `spagna_train0`  (trajectory/sequence)
3. `conf_percentile` : to remove points with lower confidence
4. `split`:  train, val, test1, test2, all


git submodule add https://github.com/bakuljangley/Depth-Anything-V2.git monocular_depth_models/depthanythingv2
git submodule add https://github.com/yourusername/another-model.git monocular_depth_models/model2

Additionally, to apply scaled pre-computed from the training set (saved in `my_vbr_utils/train_scales.json`). Run:
```
chmod+x apply_train_scales.sh
./apply_train_scales.sh
```
*both `run_evaluate.sh` and `apply_train_scales.sh` will need to be re-run for each of the 5 trajectories selected.*

## Using Monocular Depth Estimation Models as H and MASt3R as G

1. Generating Depthmaps: After setting up environments for the metric monocular depth estimation models as described [earlier](#monocular-depth-estimation-models), pre-compute depthmaps for the final anchor-query pairs.
    1. ZoeDepth: 
2. We use pre-computed depthmaps for efficiency. The script `my_scripts/evaluate_anydepth.py` allows you to add depthmaps from any source and combine them with MASt3R as G.

```
chmod+x run_evaluate_anydepth.sh 
./run_evaluate_anydepth.sh
```
*This will automatically run the script for all trajectories and does not need to be repeated.*

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