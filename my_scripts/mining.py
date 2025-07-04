## ignoring warnings from the dust3r repo (deprecated pytorch version)
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

## move one directory up to ensure imports from mast3r-v2/my_utils and original repo work
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import os
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
    parser.add_argument('--anchor_start', type=int, required=True)
    parser.add_argument('--anchor_stop', type=int, required=True)
    parser.add_argument('--anchor_step', type=int, default=50)
    parser.add_argument('--query_start', type=int, required=True)
    parser.add_argument('--query_stop', type=int, required=True)
    parser.add_argument('--query_step', type=int, default=50)
    parser.add_argument('--output', type=str, default='spagna_matches_inliers_fm.csv')
    parser.add_argument('--top_n', type=int, default=3, help='Top N anchors per query to keep')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load dataset and calibration
    dataset = vbrDataset(args.dataset, args.gt)
    calib = load_calibration(args.calib)
    from mast3r.model import AsymmetricMASt3R
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)

    anchor_indices = list(range(args.anchor_start, args.anchor_stop, args.anchor_step))
    query_indices = list(range(args.query_start, args.query_stop, args.query_step))

    results = []
    from itertools import product

    pairs = list(product(anchor_indices, query_indices))
    for anchor_idx, query_idx in tqdm(pairs, desc="Anchor-Query Pairs"):
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

            results.append({
                'anchor_idx': anchor_idx,
                'query_idx': query_idx,
                'num_matches': num_matches,
                'num_inliers': num_inliers
            })
        except Exception as e:
            tqdm.write(f"Error processing anchor {anchor_idx}, query {query_idx}: {e}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save all results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved all results to {args.output}")

    # Sort and keep top N anchors per query
    sorted_df = df.sort_values(['query_idx', 'num_inliers'], ascending=[True, False])
    topN_anchors_per_query = sorted_df.groupby('query_idx').head(args.top_n)
    topN_csv = os.path.splitext(args.output)[0] + f'_top{args.top_n}.csv'
    topN_anchors_per_query.to_csv(topN_csv, index=False)
    print(f"Saved top {args.top_n} anchors per query to {topN_csv}")

if __name__ == "__main__":
    main()