import json
import os 


def load_scene_correspondences(file_path):
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"JSON file is empty: {file_path}")
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from {file_path}: {e}")
    return data