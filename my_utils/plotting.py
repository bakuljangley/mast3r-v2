import numpy as np
import matplotlib.pyplot as plt
from .transformations import pose_to_se3
def load_pose_results(filename):
    data = np.loadtxt(filename)
    return [row for row in data if len(row) == 9]

def plot_anchor_query_estimate(ax, anchor_pose, query_pose, estimate_pose, 
                              anchor_label="Anchor", query_label="Query", est_label="Estimate",
                              anchor_color='black', query_color='green', est_color='blue'):
    # anchor_pose, query_pose, estimate_pose: (tx, ty, tz, qx, qy, qz, qw)
    def pose_to_xy(pose):
        return pose[0], pose[1]
    ax.scatter(*pose_to_xy(anchor_pose), c=anchor_color, marker='s', s=100, label=anchor_label)
    ax.scatter(*pose_to_xy(query_pose), c=query_color, marker='x', s=140, label=query_label)
    ax.scatter(*pose_to_xy(estimate_pose), c=est_color, marker='o', s=80, label=est_label)
    ax.legend()
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.axis('equal')
    ax.grid(True)

def plot_all_estimates_for_query(query_idx, dataset, results_dicts, styles):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot GT query
    query_pose = dataset[query_idx]['pose']
    T_query = pose_to_se3(query_pose)
    ax.scatter(T_query[0, 3], T_query[1, 3], c='green', marker='x', s=140, label='GT Query')
    ax.text(T_query[0, 3] + 0.5, T_query[1, 3], f"Q{query_idx}", color='green', fontsize=9)

    # Track which anchors have been plotted for legend
    plotted_anchors = set()
    legend_handles = {}

    for result_type, (by_query, style) in results_dicts.items():
        anchors = by_query.get(query_idx, [])
        for anchor_idx, pose in anchors:
            T_anchor = pose_to_se3(dataset[anchor_idx]['pose'])
            T_est = pose_to_se3(pose)

            # Plot anchor only once per anchor_idx (not per result_type)
            if anchor_idx not in plotted_anchors:
                anchor_handle = ax.scatter(
                    T_anchor[0, 3], T_anchor[1, 3],
                    c=style['anchor'], marker='s', s=100, label=f"Anchor"
                )
                plotted_anchors.add(anchor_idx)
            else:
                ax.scatter(
                    T_anchor[0, 3], T_anchor[1, 3],
                    c=style['anchor'], marker='s', s=100
                )
            # Anchor index text
            ax.text(T_anchor[0, 3] + 0.5, T_anchor[1, 3], f"A{anchor_idx}", color=style['anchor'], fontsize=9)

            # Plot estimate
            est_handle = None
            if f"{result_type} Estimate" not in legend_handles:
                est_handle = ax.scatter(
                    T_est[0, 3], T_est[1, 3],
                    c=style['estimate'], marker='o', s=80, label=result_type
                )
                legend_handles[f"{result_type} Estimate"] = est_handle
            else:
                ax.scatter(
                    T_est[0, 3], T_est[1, 3],
                    c=style['estimate'], marker='o', s=80
                )
            # Estimate index text
            ax.text(T_est[0, 3] + 0.5, T_est[1, 3], f"{anchor_idx}", color=style['estimate'], fontsize=9)

    ax.set_title(f"Estimates for Query {query_idx}")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.axis('equal')
    ax.grid(True, which='both')
    ax.minorticks_on()

    # Only show one entry per label in the legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            new_handles.append(h)
            new_labels.append(l)
            seen.add(l)
    ax.legend(new_handles, new_labels, loc='best')
    plt.tight_layout()
    plt.show()