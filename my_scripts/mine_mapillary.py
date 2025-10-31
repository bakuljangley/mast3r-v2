import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import argparse
import csv
from tqdm import tqdm
import numpy as np
import cv2
import pandas as pd

# Add MASt3R and utility imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from mast3r.model import AsymmetricMASt3R
from my_utils.mast3r_utils import get_master_output
from my_vbr_utils.vbr_dataset import vbrInterpolatedDataset


# ------------------------------ CONFIG: per-scene anchor ranges ------------------------------
SCENE_TO_SEQUENCES = {
    "spagna_train0": {
        "S0kbYi8VQ_WxGBZSOrY3Wg": [[37300, 41500, 50]],
        "n4k0JDQfdeItNMf9OOwCUQ": [
            [7781, 7822, 5],
            [23861, 23883, 5],
            [2531, 2600, 5],
        ],
        # "lLTFtBmvxOHyK7oYAUzbq9": [[37300, 41500, 50]],
    },
    "campus_train0": {
        "5kb1M1svQmCdlwVYX6iP4Q": [[11200, 11475, 10]],
    },
}
# ---------------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MASt3R matching between Mapillary queries and VBR anchors."
    )
    parser.add_argument('--anchor_scene', type=str, required=True, help="Scene name used by the VBR dataset (e.g., spagna_train0).")
    parser.add_argument('--anchor_root', type=str, required=True, help="Root folder for VBR anchors.")
    parser.add_argument('--query_root', type=str, required=True, help="Root that contains <scene>/<sequence>/ with images + metadata.csv.")
    parser.add_argument('--output_root', type=str, required=True, help="Root where results will be saved under <output_root>/<scene>/.")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_name', type=str, default="naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
    parser.add_argument('--top_n', type=int, default=3)
    return parser.parse_args()


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_scene_output_paths(output_root: str, scene: str):
    """
    Return paths inside <output_root>/<scene>/ :
      - results.csv
      - results_processed_pairs.txt
    """
    scene_dir = os.path.join(output_root, scene)
    ensure_dir(scene_dir)
    output_csv = os.path.join(scene_dir, "results.csv")
    temp_file = os.path.join(scene_dir, "results_processed_pairs.txt")
    return output_csv, temp_file


# -------- temp/resume helpers (include sequence id to avoid collisions) --------
def load_processed_pairs(temp_file: str):
    if os.path.exists(temp_file):
        print(f"[INFO] Resuming from temp file: {temp_file}")
        out = set()
        with open(temp_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                # support both new 3-field and old 2-field formats
                if len(parts) == 3:
                    out.add((parts[0], parts[1], parts[2]))  # (anchor_idx, sequence_id, query_name)
                elif len(parts) == 2:
                    out.add((parts[0], "", parts[1]))
        return out
    else:
        print(f"[INFO] Creating new temp file: {temp_file}")
        open(temp_file, 'w').close()
        return set()


def save_processed_pair(temp_file: str, anchor_idx: int, sequence_id: str, query_name: str):
    with open(temp_file, 'a') as f:
        f.write(f"{anchor_idx},{sequence_id},{query_name}\n")


def main():
    args = parse_args()

    if args.anchor_scene not in SCENE_TO_SEQUENCES:
        raise ValueError(f"[ERROR] Scene {args.anchor_scene} not found in SCENE_TO_SEQUENCES")

    sequence_anchor_ranges = SCENE_TO_SEQUENCES[args.anchor_scene]

    # VBR anchor dataset (indexed by anchor_idx)
    dataset = vbrInterpolatedDataset(args.anchor_root, args.anchor_scene)

    # Mapillary scene root: <query_root>/<scene>/
    scene_root = os.path.join(args.query_root, args.anchor_scene)
    if not os.path.isdir(scene_root):
        raise FileNotFoundError(f"[ERROR] Scene directory not found: {scene_root}")

    # Build all (anchor_idx, query_path, sequence_id) pairs
    valid_pairs = []
    for sequence_id, ranges in sequence_anchor_ranges.items():
        seq_dir = os.path.join(scene_root, sequence_id)
        if not os.path.isdir(seq_dir):
            print(f"[WARN] Missing sequence dir: {seq_dir} (skipping)")
            continue

        # images live in the sequence dir
        images = sorted(
            os.path.join(seq_dir, f)
            for f in os.listdir(seq_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not images:
            print(f"[WARN] No images in {seq_dir} (skipping)")
            continue

        # optional: filter by metadata.csv if present
        metadata_path = os.path.join(seq_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            try:
                meta_df = pd.read_csv(metadata_path)
                ids = None
                if 'id' in meta_df.columns:
                    ids = set(os.path.splitext(str(v))[0] for v in meta_df['id'].tolist())
                elif 'filename' in meta_df.columns:
                    ids = set(os.path.splitext(os.path.basename(str(v)))[0] for v in meta_df['filename'].tolist())
                if ids is not None:
                    images = [p for p in images if os.path.splitext(os.path.basename(p))[0] in ids]
            except Exception as e:
                print(f"[WARN] Could not read metadata in {seq_dir}: {e}")

        # add all anchor/query combinations for this sequence
        for q_path in images:
            for start, stop, step in ranges:
                for anchor_idx in range(start, stop, step):
                    valid_pairs.append((anchor_idx, q_path, sequence_id))

    print(f"[INFO] Total anchor–query pairs to evaluate: {len(valid_pairs)}")
    if not valid_pairs:
        raise RuntimeError("No valid pairs found. Check SCENE_TO_SEQUENCES and folder layout.")

    # Load MASt3R once
    model = AsymmetricMASt3R.from_pretrained(args.model_name).to(args.device)

    # Prepare outputs under <output_root>/<scene>/
    output_csv, temp_file = get_scene_output_paths(args.output_root, args.anchor_scene)
    processed_pairs = load_processed_pairs(temp_file)

    fieldnames = ['scene', 'sequence', 'query_image', 'anchor_idx', 'num_matches', 'num_inliers']
    csv_exists = os.path.exists(output_csv)

    with open(output_csv, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not csv_exists or os.stat(output_csv).st_size == 0:
            writer.writeheader()

        for anchor_idx, query_path, sequence_id in tqdm(valid_pairs, desc="Matching"):
            query_name = os.path.splitext(os.path.basename(query_path))[0]

            if (str(anchor_idx), sequence_id, query_name) in processed_pairs:
                continue

            try:
                anchor = dataset[anchor_idx]  # dict with 'image' key

                out_aq = get_master_output(model, args.device, anchor['image'], query_path,
                                           visualize=False, verbose=False)
                out_qa = get_master_output(model, args.device, query_path, anchor['image'],
                                           visualize=False, verbose=False)

                m0_aq, m1_aq = out_aq[0], out_aq[1]
                m0_qa, m1_qa = out_qa[0], out_qa[1]

                # ensure numpy float32 for OpenCV
                m0_aq = np.asarray(m0_aq, dtype=np.float32)
                m1_aq = np.asarray(m1_aq, dtype=np.float32)
                m0_qa = np.asarray(m0_qa, dtype=np.float32)
                m1_qa = np.asarray(m1_qa, dtype=np.float32)

                if len(m0_aq) >= 8 and len(m0_qa) >= 8:
                    _, mask_aq = cv2.findFundamentalMat(m0_aq, m1_aq, cv2.FM_RANSAC, 1.0, 0.99)
                    _, mask_qa = cv2.findFundamentalMat(m0_qa, m1_qa, cv2.FM_RANSAC, 1.0, 0.99)
                    inliers = min(int(mask_aq.sum() if mask_aq is not None else 0),
                                  int(mask_qa.sum() if mask_qa is not None else 0))
                    matches = min(len(m0_aq), len(m0_qa))
                else:
                    inliers = 0
                    matches = 0

                writer.writerow({
                    "scene": args.anchor_scene,
                    "sequence": sequence_id,
                    "query_image": f"{sequence_id}/{query_name}",
                    "anchor_idx": anchor_idx,
                    "num_matches": matches,
                    "num_inliers": inliers
                })
                csvfile.flush()
                save_processed_pair(temp_file, anchor_idx, sequence_id, query_name)

            except Exception as e:
                print(f"[Error] Anchor {anchor_idx} x {sequence_id}/{query_name}: {e}")

    print(f"[INFO] Saved raw results to {output_csv}")

    # Top-N anchors per query
    df = pd.read_csv(output_csv)
    if not df.empty:
        df_sorted = df.sort_values(['query_image', 'num_inliers'], ascending=[True, False])
        topN = df_sorted.groupby('query_image', as_index=False).head(args.top_n)
        topN_csv = output_csv.replace(".csv", f"_top{args.top_n}_anchors_per_query.csv")
        topN.to_csv(topN_csv, index=False)
        print(f"[INFO] Saved top {args.top_n} anchors per query to {topN_csv}")
    else:
        print("[WARN] Results CSV is empty; skipping top-N export.")


if __name__ == "__main__":
    main()
