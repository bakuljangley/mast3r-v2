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
from itertools import product

# Import your dataset and model utilities
from my_utils.my_vbr_dataset import vbrDataset, load_calibration
from my_utils.mast3r_utils import get_master_output

def parse_args():
    parser = argparse.ArgumentParser(description="Mine anchor-query pairs with inlier counting.")
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset in kitti format')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth file')
    parser.add_argument('--calib', type=str, required=True, help='Path to calibration file')
    parser.add_argument('--sequence_pairs', type=str, required=True, help='Anchor-query sequence pairs in the format \"anchor1:query1,query2;anchor2:query3,query4\"')
    parser.add_argument('--anchor_step', type=int, default=50)
    parser.add_argument('--query_step', type=int, default=50)
    parser.add_argument('--output', type=str, default='spagna_matches_inliers_fm.csv')
    parser.add_argument('--top_n', type=int, default=3, help='Top N anchors per query to keep')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument('--temp_file', type=str, default='processed_pairs.txt', help='File to track processed pairs')
    return parser.parse_args()

def parse_sequence_pairs(sequence_pairs_str):
    """
    Parse the sequence pairs argument into a dictionary.
    Format: "anchor1:query1,query2;anchor2:query3,query4"
    Returns: {anchor1: [query1, query2], anchor2: [query3, query4]}
    """
    sequence_pairs = {}
    pairs = sequence_pairs_str.split(';')
    for pair in pairs:
        anchor, queries = pair.split(':')
        sequence_pairs[int(anchor)] = [int(q) for q in queries.split(',')]
    return sequence_pairs

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
    dataset = vbrDataset(args.dataset, args.gt)
    calib = load_calibration(args.calib)
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)

    # Parse sequence pairs
    sequence_pairs = parse_sequence_pairs(args.sequence_pairs)

    # Ensure the directory for the temporary file exists
    ensure_dir_exists(args.temp_file)
    if not os.path.exists(args.temp_file):
        open(args.temp_file, 'w').close()
    processed_pairs = load_processed_pairs(args.temp_file)

    # Ensure output directory exists
    ensure_dir_exists(args.output)
    output_exists = os.path.exists(args.output)
    fieldnames = ['anchor_seq', 'query_seq','anchor_idx', 'query_idx', 'num_matches', 'num_inliers']

    # Open output CSV for appending
    with open(args.output, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not output_exists:
            writer.writeheader()

        # Process each anchor-query sequence pair
        for anchor_seq, query_seqs in sequence_pairs.items():
            anchor_indices = list(range(anchor_seq * 2000, (anchor_seq + 1) * 2000, args.anchor_step))
            for query_seq in query_seqs:
                query_indices = list(range(query_seq * 2000, (query_seq + 1) * 2000, args.query_step))
                pairs = list(product(anchor_indices, query_indices))
                for anchor_idx, query_idx in tqdm(pairs, desc=f"Anchor {anchor_seq} - Query {query_seq}"):
                    # Skip already processed pairs
                    if (anchor_idx, query_idx) in processed_pairs:
                        continue

                    try:
                        anchor = dataset[anchor_idx]
                        query = dataset[query_idx]
                        output = get_master_output(
                            model, args.device,
                            anchor['image_left'], query['image_left'],
                            visualize=False, verbose=False
                        )
                        matches_im0 = output[0]
                        matches_im1 = output[1]
                        num_matches = len(matches_im0)

                        if num_matches >= 8:
                            F, mask = cv2.findFundamentalMat(matches_im0, matches_im1, cv2.FM_RANSAC, 1.0, 0.99)
                            num_inliers = int(mask.sum()) if mask is not None else 0
                        else:
                            num_inliers = 0

                        # Save result immediately
                        writer.writerow({
                            'anchor_seq': anchor_seq,
                            'query_seq': query_seq,
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

    # Sort so the highest inliers come first for each (query_idx, anchor_seq)
    sorted_df = df.sort_values(['query_idx', 'anchor_seq', 'num_inliers'], ascending=[True, True, False])

    # For each (query_idx, anchor_seq), keep the top N anchors (anchor_idx) with the most inliers
    topN_anchors_per_query_per_anchorseq = sorted_df.groupby(['query_idx', 'anchor_seq']).head(args.top_n)

    # Save to CSV
    topN_csv = os.path.splitext(args.output)[0] + f'_top{args.top_n}_anchors_per_query_per_anchorseq.csv'
    topN_anchors_per_query_per_anchorseq.to_csv(topN_csv, index=False)
    print(f"Saved top {args.top_n} anchors per query per anchor sequence to {topN_csv}")


if __name__ == "__main__":
    main()

