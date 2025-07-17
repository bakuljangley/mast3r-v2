## ignoring warnings from the dust3r repo (deprecated pytorch version)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

## move one directory up to ensure imports from mast3r-v2/my_utils and original repo work
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import csv
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import json
# Import your dataset and model utilities
from my_vbr_utils.vbr_dataset import VBRDataset, load_scene_calibration
from my_utils.mast3r_utils import get_master_output

CONFIG_PATH = "/home/bjangley/VPR/mast3r-v2/my_vbr_utils/vbrPaths.yaml"

def parse_args():
    parser = argparse.ArgumentParser(description="Mine anchor-query pairs with inlier counting.")
    parser.add_argument('--dataset_scene', type=str, required=True, help='Path to dataset in kitti format')
    parser.add_argument('--anchor_query_json', type=str, required=True, help='Anchor-query sequence ranges"')
    parser.add_argument('--anchor_step', type=int, default=50)
    parser.add_argument('--query_step', type=int, default=50)
    parser.add_argument('--output', type=str, default='spagna_matches_inliers_fm.csv')
    parser.add_argument('--top_n', type=int, default=3, help='Top N anchors per query to keep')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument('--temp_file', type=str, default='processed_pairs.txt', help='File to track processed pairs')
    return parser.parse_args()

def load_anchor_query_dict(json_file_path):
    """
    Load the anchor-query dictionary from a JSON file.
    """
    with open(json_file_path, 'r') as f:
        loaded_dict = json.load(f)

    # Convert string keys back to tuples
    anchor_query_dict = {tuple(map(int, key.strip("()").split(","))): value for key, value in loaded_dict.items()}
    return anchor_query_dict

def generate_pairs_from_ranges(anchor_query_dict, anchor_step, query_step):
    """
    Generate anchor-query index pairs from anchor and query ranges.
    """
    all_pairs = []
    for anchor_range, query_ranges in anchor_query_dict.items():
        anchor_indices = list(range(anchor_range[0], anchor_range[1], anchor_step))
        for query_range in query_ranges:
            query_indices = list(range(query_range[0], query_range[1], query_step))
            pairs = [(a, q) for a in anchor_indices for q in query_indices]
            all_pairs.extend(pairs)
    return all_pairs

def load_processed_pairs(temp_file):
    """
    Load processed pairs from the temporary file.
    Returns: Set of processed pairs (anchor_idx, query_idx).
    """
    if os.path.exists(temp_file):
        with open(temp_file, 'r') as f:
            return set(tuple(map(int, line.strip().split(','))) for line in f)
    return set()

def save_processed_pair(temp_file, anchor_idx, query_idx):
    """
    Save a processed pair to the temporary file.
    """
    with open(temp_file, 'a') as f:
        f.write(f"{anchor_idx},{query_idx}\n")

def ensure_dir_exists(path):
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)

def main():
    args = parse_args()

    # Load dataset and calibration
    all_loaded = VBRDataset(CONFIG_PATH, locations=[args.dataset_scene])
    dataset = all_loaded.get_combined_dataset()
    calib = load_scene_calibration(location_name=args.dataset_scene, config_path=CONFIG_PATH)
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)

    # Load anchor query dictionary from JSON
    anchor_query_dict = load_anchor_query_dict(args.anchor_query_json)

    # Generate anchor-query pairs
    all_pairs = generate_pairs_from_ranges(anchor_query_dict, args.anchor_step, args.query_step)

    # Ensure the directory for the temporary file exists
    ensure_dir_exists(args.temp_file)
    if not os.path.exists(args.temp_file):
        open(args.temp_file, 'w').close()
    processed_pairs = load_processed_pairs(args.temp_file)

    # Ensure output directory exists
    ensure_dir_exists(args.output)
    output_exists = os.path.exists(args.output)
    fieldnames = ['anchor_idx', 'query_idx', 'num_matches', 'num_inliers']

    # Open output CSV for appending
    with open(args.output, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not output_exists:
            writer.writeheader()

        # Process each anchor-query pair
        for anchor_idx, query_idx in tqdm(all_pairs, desc="Processing pairs"):
            # Skip already processed pairs
            if (anchor_idx, query_idx) in processed_pairs:
                continue

            try:
                anchor = dataset[anchor_idx]
                query = dataset[query_idx]
                #first image is anchor
                output_aq = get_master_output(
                    model, args.device,
                    anchor['image'], query['image'],
                    visualize=False, verbose=False
                )
                matches_im0_aq = output_aq[0]
                matches_im1_aq = output_aq[1]
                num_matches_aq = len(matches_im0_aq)
                if num_matches_aq >= 8:
                    F_aq, mask_aq = cv2.findFundamentalMat(matches_im0_aq, matches_im1_aq, cv2.FM_RANSAC, 1.0, 0.99)
                    num_inliers_aq = int(mask_aq.sum()) if mask_aq is not None else 0
                else:
                    num_inliers_aq = 0
                #first image is query
                output_qa = get_master_output(
                    model, args.device,
                    query['image'], anchor['image'],
                    visualize=False, verbose=False
                )
                matches_im0_qa = output_qa[0]
                matches_im1_qa = output_qa[1]
                num_matches_qa = len(matches_im0_qa)
                if num_matches_qa >= 8:
                    F_qa, mask_qa = cv2.findFundamentalMat(matches_im0_qa, matches_im1_qa, cv2.FM_RANSAC, 1.0, 0.99)
                    num_inliers_qa = int(mask_qa.sum()) if mask_qa is not None else 0
                else:
                    num_inliers_qa = 0

                num_inliers = min(num_inliers_aq, num_inliers_qa)
                num_matches = min(num_matches_aq, num_matches_qa)

                # Find the anchor and query ranges
                anchor_range = next((k for k, v in anchor_query_dict.items() if k[0] <= anchor_idx < k[1]), None)
                query_range = next((qr for k, v in anchor_query_dict.items() for qr in v if qr[0] <= query_idx < qr[1]), None)

                # Save result immediately
                writer.writerow({
                    'anchor_idx': anchor_idx,
                    'query_idx': query_idx,
                    'num_matches': num_matches,
                    'num_inliers': num_inliers
                })
                csvfile.flush()  # ensure data is written

                # Save processed pair
                save_processed_pair(args.temp_file, anchor_idx, query_idx)

            except Exception as e:
                tqdm.write(f"Error processing anchor {anchor_idx}, query {query_idx}: {e}")

    print(f"Saved all results to {args.output}")
    df = pd.read_csv(args.output)
    # Add anchor_range column
    df['anchor_range'] = None
    for k, v in anchor_query_dict.items():
        df.loc[(df['anchor_idx'] >= k[0]) & (df['anchor_idx'] < k[1]), 'anchor_range'] = str(k)

    # Sort so the highest inliers come first for each query_idx and anchor_range
    sorted_df = df.sort_values(['query_idx', 'anchor_range', 'num_inliers'], ascending=[True, True, False])

    # For each query_idx and anchor_range, keep the top N anchors (anchor_idx) with the most inliers
    topN_anchors_per_query = sorted_df.groupby(['query_idx', 'anchor_range']).head(args.top_n)

    # Save to CSV
    topN_csv = os.path.splitext(args.output)[0] + f'_top{args.top_n}_anchors_per_query.csv'
    topN_anchors_per_query.to_csv(topN_csv, index=False)
    print(f"Saved top {args.top_n} anchors per query to {topN_csv}")


if __name__ == "__main__":
    main()