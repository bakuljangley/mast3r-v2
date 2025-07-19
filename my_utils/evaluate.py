import pandas as pd

def load_statistics(pointcloud_type, base_path, scene_name):
    """
    Load statistics for a specified point cloud type.
    Auto-detects column structure from the data.
    """
    statistics_file = f"{base_path}/{scene_name}_{pointcloud_type}_statistics.txt"
    try:
        df_temp = pd.read_csv(statistics_file, delim_whitespace=True, header=None)
        num_columns = len(df_temp.columns)

        # Define column names based on detected structure
        if num_columns == 13:
            columns = [
                "query_idx", "anchor_idx",
                "num_matches", "num_inliers", "num_overlapping", "median_depth",
                "x_error", "y_error", "z_error", "pos_error", "rot_error", 
                "distance_anchor_query", "status"
            ]
        elif num_columns == 14:
            columns = [
                "query_idx", "anchor_idx",
                "num_matches", "num_inliers", "num_overlapping", "median_depth",
                "x_error", "y_error", "z_error", "pos_error", "rot_error", 
                "distance_anchor_query", "pointmap_error", "status"
            ]            
        elif num_columns == 15:
            columns = [
                "query_idx", "anchor_idx",
                "num_matches", "num_inliers", "num_overlapping", "median_depth",
                "x_error", "y_error", "z_error", "pos_error", "rot_error", 
                "distance_anchor_query", "pointmap_error", "scale", "status"
            ]
        elif num_columns == 17:
            columns = [
                "query_idx", "anchor_idx",
                "num_matches", "num_inliers", "num_overlapping", "median_depth",
                "x_error", "y_error", "z_error", "pos_error", "rot_error", 
                "distance_anchor_query", "pointmap_error", 
                "scale_x", "scale_y", "scale_z", "status"
            ]
        else:
            raise ValueError(f"Unexpected number of columns ({num_columns}) for {pointcloud_type}")

        # Reload with proper column names
        df = pd.read_csv(statistics_file, delim_whitespace=True, names=columns)
        print(f"Loaded statistics for {pointcloud_type}: {num_columns} columns detected")
        return df
    except FileNotFoundError:
        print(f"Error: File not found for point cloud type: {pointcloud_type}")
        return None
    

