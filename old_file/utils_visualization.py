import math
import os
import re
import pickle

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    LogNorm,
    PowerNorm,
)
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler

from old_file.utils_testing import *


####################################################################################################
####################################################################################################

#   ╔═════════════════════╗
#   ║   Process Results   ║
#   ╚═════════════════════╝


def compute_metrics(episodes):
    """
    Compute aggregate metrics from a list of episodes.

    This function processes multiple episodes and computes summary statistics 
    related to performance, activations of the uncertainty filter (UF) and 
    Control Barrier Function (CBF), confusion-matrix-derived rates, velocities, 
    and goal-related measures.

    Parameters
    ----------
    episodes : list of dict
        List of episode-level statistics, typically produced by `extract_stats`.

    Returns
    -------
    dict
        Dictionary of computed metrics including:
            - reward_mean : float
                Mean total reward across episodes.
            - length_mean : float
                Mean episode length.
            - collisions_mean : float
                Mean number of collisions per episode.
            - success_rate : float
                Proportion of successful episodes.
            - uf_activation_prop : float
                Average proportion of steps where the UF was active.
            - cbf_activation_prop : float
                Average proportion of steps where the CBF was active.
            - true_positive_rate : float
                Proportion of steps correctly identified as requiring CBF.
            - false_positive_rate : float
                Proportion of steps where UF activated unnecessarily.
            - true_negative_rate : float
                Proportion of steps correctly identified as safe.
            - false_negative_rate : float
                Proportion of steps where UF failed to activate despite CBF being active.
            - forward_velocity_mean : float
                Mean forward velocity across episodes.
            - lateral_velocity_mean : float
                Mean lateral velocity across episodes.
            - velocity_mean : float
                Combined mean of forward and lateral velocities.
            - goal_approach_mean : float
                Mean distance to goal across episodes.
            - ema_divergence_mean : float
                Mean distance to the Exponential Moving Average (EMA) center across episodes.
    """

    n_eps = len(episodes)
    if n_eps == 0:
        return {}

    # Episode-level metrics
    mean_reward = np.mean([ep['total_reward'] for ep in episodes])
    mean_length = np.mean([ep['total_length'] for ep in episodes])
    mean_collisions = np.mean([ep['total_collisions'] for ep in episodes])
    success_rate = np.mean([ep['total_success'] for ep in episodes])

    # Activation metrics (UF, CBF, confusion matrix)
    uf_props = []
    cbf_props = []
    tp = fp = tn = fn = 0
    total_steps = 0

    for ep in episodes:
        uf = np.array(ep['uf_activation'], dtype=bool)
        cbf = np.array(ep['cbf_activation_avg'], dtype=float) > 0
        steps = len(uf)

        if steps == 0:
            continue

        # Proportion of activations per episode
        uf_props.append(np.mean(uf))
        cbf_props.append(np.mean(cbf))

        # Confusion matrix counts
        tp += np.sum(uf & cbf)
        fp += np.sum(uf & ~cbf)
        tn += np.sum(~uf & ~cbf)
        fn += np.sum(~uf & cbf)
        total_steps += steps

    uf_activation_prop = np.mean(uf_props) if uf_props else 0
    cbf_activation_prop = np.mean(cbf_props) if cbf_props else 0

    # Rates derived from confusion matrix
    if total_steps > 0:
        if (tp + fn) > 0:
            tp_rate = tp / (tp + fn)   # recall / sensitivity
            fn_rate = fn / (tp + fn)
        else:
            tp_rate = fn_rate = 0

        if (fp + tn) > 0:
            fp_rate = fp / (fp + tn)   # false positive rate
            tn_rate = tn / (fp + tn)   # specificity
        else:
            fp_rate = tn_rate = 0
    else:
        tp_rate = fp_rate = tn_rate = fn_rate = 0

    # Velocity metrics
    mean_forward_velocity = np.mean([
        np.mean(ep['f_velocity']) if ep['f_velocity'] else 0 for ep in episodes
    ])
    mean_lateral_velocity = np.mean([
        np.mean(ep['l_velocity'] + ep['r_velocity'])/2 if (ep['l_velocity'] and ep['r_velocity']) else 0
        for ep in episodes
    ])
    mean_velocity = np.mean([mean_forward_velocity, mean_lateral_velocity])

    # Goal-related metrics
    mean_goal_approach = np.mean([
        np.mean(ep['dist_goal']) if ep['dist_goal'] else 0 for ep in episodes
    ])
    mean_ema_divergence = np.mean([
        np.mean(ep['dist_ema']) if ep['dist_ema'] else 0 for ep in episodes
    ])

    return {
        "reward_mean": mean_reward,
        "length_mean": mean_length,
        "collisions_mean": mean_collisions,
        "success_rate": success_rate,

        "uf_activation_prop": uf_activation_prop,
        "cbf_activation_prop": cbf_activation_prop,

        "true_positive_rate": tp_rate,
        "false_positive_rate": fp_rate,
        "true_negative_rate": tn_rate,
        "false_negative_rate": fn_rate,

        "forward_velocity_mean": mean_forward_velocity,
        "lateral_velocity_mean": mean_lateral_velocity,
        "velocity_mean": mean_velocity,

        "goal_approach_mean": mean_goal_approach,
        "ema_divergence_mean": mean_ema_divergence
    }


def extract_number(pattern, text, default=-1):
    """
    Extract the first numeric value from a string using a regex pattern.

    The function searches for the first match of the provided regular expression 
    in the given text. If a match is found, it attempts to convert the first 
    captured group into a float. If no match is found or conversion fails, 
    a default value is returned.

    Parameters
    ----------
    pattern : str
        Regular expression with at least one capturing group for the number.
    text : str
        Input string to search.
    default : float, optional
        Value returned if no match is found or conversion fails (default is -1).

    Returns
    -------
    float
        Extracted numeric value if successful, otherwise the default.
    """

    # Search for the first match of the regex pattern in text
    match = re.search(pattern, text)
    if match:
        try:
            # Convert the first captured group to float
            return float(match.group(1))
        except ValueError:
            return default
    # Return default if no match or conversion fails
    return default

def compute_metrics_for_folder(folder_path, compute_fn, output_folder):
    """
    Compute metrics for all experiments stored as pickle files in a folder.

    This function iterates over all `.pkl` files in the specified folder, 
    extracts episode statistics, computes metrics (using a user-provided 
    function), and aggregates them into DataFrames. Metrics are computed 
    for all episodes, successful episodes, and failed episodes separately. 
    Results are saved as CSV files in the output folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing experiment `.pkl` files.
    compute_fn : callable
        Function that takes a list of episodes and returns a dictionary 
        of computed metrics (e.g., `compute_metrics`).
    output_folder : str
        Directory where the aggregated CSV files will be saved.

    Returns
    -------
    tuple of pandas.DataFrame
        - df_all : DataFrame with metrics for all episodes.
        - df_success : DataFrame with metrics for successful episodes only.
        - df_failure : DataFrame with metrics for failed episodes only.

    Outputs
    -------
    {output_folder}/{folder_name}_all.csv : CSV
        Metrics for all episodes in the folder.
    {output_folder}/{folder_name}_success.csv : CSV
        Metrics for successful episodes only.
    {output_folder}/{folder_name}_failure.csv : CSV
        Metrics for failed episodes only.
    """

    results_all = []
    results_success = []
    results_failure = []

    for fname in os.listdir(folder_path):
        if fname.endswith(".pkl"):
            fpath = os.path.join(folder_path, fname)
            with open(fpath, "rb") as f:
                data = pickle.load(f)
                episodes = data["stats"]

            # Extract config info from filename
            cbf_config = extract_number(r'_cbf(\d+(?:\.\d+)?)_', fname)
            percentile = extract_number(r'_(\d+(?:\.\d+)?)pctl', fname, default=-1)

            # All episodes
            metrics_all = compute_fn(episodes)
            metrics_all["cbf_config"] = cbf_config
            metrics_all["percentile"] = percentile
            results_all.append(metrics_all)

            # Successful episodes only
            episodes_success = [ep for ep in episodes if ep["total_success"] == 1]
            if episodes_success:
                metrics_success = compute_fn(episodes_success)
                metrics_success["cbf_config"] = cbf_config
                metrics_success["percentile"] = percentile
                results_success.append(metrics_success)

            # Failed episodes only
            episodes_failure = [ep for ep in episodes if ep["total_success"] == 0]
            if episodes_failure:
                metrics_failure = compute_fn(episodes_failure)
                metrics_failure["cbf_config"] = cbf_config
                metrics_failure["percentile"] = percentile
                results_failure.append(metrics_failure)

    # Build DataFrames and sort by config and percentile
    df_all = pd.DataFrame(results_all).sort_values(by=["cbf_config", "percentile"]).reset_index(drop=True)
    df_success = pd.DataFrame(results_success).sort_values(by=["cbf_config", "percentile"]).reset_index(drop=True)
    df_failure = pd.DataFrame(results_failure).sort_values(by=["cbf_config", "percentile"]).reset_index(drop=True)

    # Save CSVs with folder name prefix
    folder_name = os.path.basename(os.path.normpath(folder_path))
    os.makedirs(output_folder, exist_ok=True)

    df_all.to_csv(os.path.join(output_folder, f"{folder_name}_all.csv"), index=False)
    df_success.to_csv(os.path.join(output_folder, f"{folder_name}_success.csv"), index=False)
    df_failure.to_csv(os.path.join(output_folder, f"{folder_name}_failure.csv"), index=False)

    return df_all, df_success, df_failure


def process_all_subfolders(root_folder, compute_fn, output_folder):
    """
    Process all subfolders within a root directory and compute metrics.

    This function iterates over all subfolders of the given root directory, 
    and for each subfolder it applies `compute_metrics_for_folder` to compute 
    episode-level metrics. Results are saved as CSV files in the specified 
    output folder.

    Parameters
    ----------
    root_folder : str
        Path to the root directory containing subfolders of experiment results.
    compute_fn : callable
        Function that takes a list of episodes and returns a dictionary 
        of computed metrics (e.g., `compute_metrics`).
    output_folder : str
        Directory where the aggregated CSV files will be saved.

    Returns
    -------
    None
        Results are saved as CSV files in the `output_folder`.
    """

    for sub in os.listdir(root_folder):
        sub_path = os.path.join(root_folder, sub)

        # Check if entry is a folder
        if os.path.isdir(sub_path):
            print(f"Processing {sub_path}...")

            # Apply metrics computation to this subfolder
            compute_metrics_for_folder(sub_path, compute_fn, output_folder)

####################################################################################################
####################################################################################################

#   ╔═══════════════════════════╗
#   ║   Results Visualization   ║
#   ╚═══════════════════════════╝


metric_labels = {
    "reward_mean": "Average cumulaitive reward",
    "length_mean": "Average episode length",
    "collisions_mean": "Average collisions",
    "success_rate": "Success rate",

    "uf_activation_prop": "UF activation proportion",
    "cbf_activation_prop": "CBF activation proportion",

    "true_positive_rate": "True positives",
    "false_positive_rate": "False positives",
    "true_negative_rate": "True negatives",
    "false_negative_rate": "False negatives",

    "forward_velocity_mean": "Forward velocity",
    "velocity_mean": "Average velocity",

    "goal_approach_mean": "Goal approaching",
    "ema_divergence_mean": "EMA distancing",
}

def plot_metrics(
    path_list,
    labels,
    cbf_config=1,
    percentiles=[1, 10, 20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 99],
    upper_bound=None,
    lower_bound=None,
    ncols=1,
    logy=False,
    plot_std=False,
    metrics=None,
    title=None,
    fontsize=16,
    legend=True
):
    """
    Plot evaluation metrics across percentiles for one or more experiments.

    This function loads CSV result files, filters them by the given 
    `cbf_config` and percentiles, and generates line plots for selected metrics. 
    Optionally, it overlays standard deviation bands, reference bounds, 
    and customizes the layout and appearance.

    Parameters
    ----------
    path_list : list of str
        Paths to CSV result files.
    labels : list of str
        Labels corresponding to each path (used in the legend).
    cbf_config : int, optional
        CBF configuration index to filter results (default is 1).
    percentiles : list of int, optional
        List of percentiles to include on the x-axis (default is [1, 10, ..., 99]).
    upper_bound : dict, optional
        Dictionary mapping metric names to upper reference values (drawn as dashed red lines).
    lower_bound : dict, optional
        Dictionary mapping metric names to lower reference values (drawn as dotted blue lines).
    ncols : int, optional
        Number of columns in the subplot grid (default is 1).
    logy : bool, optional
        If True, y-axis is shown on a logarithmic scale (default is False).
    plot_std : bool, optional
        If True, plot shaded bands representing ±1 standard deviation (default is False).
    metrics : list of str, optional
        List of metric names to plot. Must be present as columns in the CSVs.
    title : str, optional
        Global title for the figure (default is None).
    fontsize : int, optional
        Base font size for labels, ticks, and legends (default is 16).
    legend : bool, optional
        If True, show legends for each subplot (default is True).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plotted subplots.
    """

    # Load CSVs and filter by percentiles and cbf_config
    dfs = []
    for p in path_list:
        df = pd.read_csv(p)
        df = df.loc[(df["percentile"].isin(percentiles)) & (df["cbf_config"] == cbf_config)]
        dfs.append(df)

    # Metrics for which we skip drawing reference bounds
    skip_bounds = {
        "uf_activation_prop",
        "cbf_activation_prop",

        "true_positive_rate",
        "false_positive_rate",
        "true_negative_rate",
        "false_negative_rate",

        "forward_velocity_mean",
        "velocity_mean",

        "goal_approach_mean",
        "ema_divergence_mean"
    }

    # Prepare figure and axes
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 4 * nrows))

    # Ensure axes is a flat array
    axes = np.atleast_1d(axes).flatten()

    # Loop over metrics to plot
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # Plot one curve per dataframe
        for j, (df, label) in enumerate(zip(dfs, labels)):
            color = "grey" if j == len(dfs) - 1 else None
            line, = ax.plot(df["percentile"], df[metric], marker="o", label=label, color=color)

            # Add std band if available
            if plot_std:
                std_col = metric.replace("_mean", "_std")
                if std_col in df:
                    ax.fill_between(
                        df["percentile"],
                        df[metric] - df[std_col],
                        df[metric] + df[std_col],
                        alpha=0.2,
                        color=line.get_color()
                    )

        # Add reference bounds if defined for this metric
        if metric not in skip_bounds:
            if metric in upper_bound:
                ax.axhline(
                    y=upper_bound[metric],
                    color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="policy"
                )
            if metric in lower_bound:
                ax.axhline(
                    y=lower_bound[metric],
                    color="blue", linestyle=":", linewidth=1.5, alpha=0.7, label="policy+cbf"
                )

        # Axis labels
        ax.set_xlabel("tau", fontsize=fontsize)
        ax.set_ylabel(metric_labels[metric], fontsize=fontsize)

        # Tick label size
        ax.tick_params(axis="both", which="major", labelsize=fontsize)

        # Log scale option
        if logy:
            ax.set_yscale("log")

        # Legend
        if legend:
            ax.legend(loc="best", fontsize=fontsize - 2)

    # Disable unused subplots
    for ax in axes[len(metrics):]:
        ax.axis("off")

    # Global title if provided
    if title:
        ax.set_title(title, fontsize=fontsize)
            
    plt.tight_layout()
    return fig


def plot_metric_series_multi(
    path_list_x,
    path_list_y,
    metric_x,
    metric_y,
    labels,
    cmap_name="plasma",
    cmap_col="percentile",
    upper_bound_x=None,
    upper_bound_y=None,
    lower_bound_x=None,
    lower_bound_y=None,
    ncols=2,
    title=None,
    fontsize=14
):
    """
    Plot paired metric trajectories across percentiles for multiple experiments.

    This function generates 2D plots where each curve shows the trajectory of
    two metrics (X vs Y) across percentiles. Curves are color-coded by the 
    percentile value, and a baseline (random UE) is included for comparison. 
    Optional reference bounds for "policy" and "policy+cbf" can be drawn.

    Parameters
    ----------
    path_list_x : list of str
        Paths to CSV files containing the X-axis metric values.
    path_list_y : list of str
        Paths to CSV files containing the Y-axis metric values.
    metric_x : str
        Column name of the metric to plot on the x-axis.
    metric_y : str
        Column name of the metric to plot on the y-axis.
    labels : list of str
        Labels for each experiment curve (used in the legend).
    cmap_name : str, optional
        Name of the colormap used to color curves by percentile (default is "plasma").
    cmap_col : str, optional
        Column name in the CSV that encodes percentile values (default is "percentile").
    upper_bound_x : dict, optional
        Dictionary mapping metric names to x-values for the "policy" reference point.
    upper_bound_y : dict, optional
        Dictionary mapping metric names to y-values for the "policy" reference point.
    lower_bound_x : dict, optional
        Dictionary mapping metric names to x-values for the "policy+cbf" reference point.
    lower_bound_y : dict, optional
        Dictionary mapping metric names to y-values for the "policy+cbf" reference point.
    ncols : int, optional
        Number of columns in the subplot grid (default is 2).
    title : str, optional
        Global title for the figure (default is None).
    fontsize : int, optional
        Base font size for labels, ticks, and legends (default is 14).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plotted subplots.
    """


    def truncate_cmap(cmap, vmin=0.1, vmax=0.9, n=256):
        # Restrict colormap range to improve contrast
        return LinearSegmentedColormap.from_list(
            f"trunc({cmap.name},{vmin:.2f},{vmax:.2f})",
            cmap(np.linspace(vmin, vmax, n))
        )
        
    # Load dataframes
    data_x, data_y = [], []
    for path_x, path_y in zip(path_list_x, path_list_y):
        data_x.append(pd.read_csv(path_x))
        data_y.append(pd.read_csv(path_y))

    # Prepare subplots
    n_subplots = len(path_list_y) - 1
    nrows = int(np.ceil(n_subplots / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).flatten()

    cmap = truncate_cmap(plt.cm.get_cmap(cmap_name), 0.1, 0.85)
    
    # Loop over datasets (last one is baseline)
    for i in range(len(path_list_x[:-1])):
        ax = axes[i]
        
        df_x, df_y = data_x[i], data_y[i]
        
        # Extract values
        x_vals = df_x[metric_x].values
        y_vals = df_y[metric_y].values
        perc_vals = df_x[cmap_col].values

        # Colored line segments (interpolated by tau)
        points = np.column_stack([x_vals, y_vals]).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_vals = (perc_vals[:-1] + perc_vals[1:]) / 2

        lc = LineCollection(segments, array=seg_vals, cmap=cmap, linewidths=2, alpha=0.8, zorder=1)
        ax.add_collection(lc)

        # Scatter plot of points with tau coloring
        sc = ax.scatter(
            x_vals, y_vals,
            c=perc_vals, cmap=cmap,
            s=50, edgecolors="black", linewidths=0.4, zorder=2
        )

        # Grey baseline (random UE)
        base_line, = ax.plot(
            data_x[-1][metric_x], data_y[-1][metric_y],
            color="0.6", linestyle="-", linewidth=1, alpha=0.9, zorder=0, label="random_UE"
        )
        ax.scatter(
            data_x[-1][metric_x], data_y[-1][metric_y],
            color="0.5", s=50, edgecolors="black", linewidths=0.5, zorder=1
        )

        # Legend handles
        handles = []

        # Colored curve (median tau color for legend)
        norm_val = (np.median(perc_vals) - perc_vals.min()) / (perc_vals.max() - perc_vals.min())
        median_color = cmap(norm_val)
        handles.append(Line2D([0], [0], color=median_color, linewidth=2, label=labels[i]))
        handles.append(base_line)

        # Optional dashed line between policy and policy+cbf bounds
        if upper_bound_x and upper_bound_y:
            ax.plot(
                [upper_bound_x[metric_x], lower_bound_x[metric_x]],
                [upper_bound_y[metric_y], lower_bound_y[metric_y]],
                color="0.6", linestyle="--", linewidth=1, alpha=0.9, zorder=0
            )

            # Blue "x" marker for policy
            ax.scatter(
                upper_bound_x[metric_x], upper_bound_y[metric_y],
                color="blue", marker="x", s=80, linewidths=2, zorder=2
            )

            # Green "x" marker for policy+cbf
            ax.scatter(
                lower_bound_x[metric_x], lower_bound_y[metric_y],
                color="green", marker="x", s=80, edgecolors="black", linewidths=2, zorder=2
            )

            # Add reference line and markers to legend
            handles.append(Line2D([0], [0], color="0.6", linestyle="--", linewidth=2, label="policy → policy+cbf"))
            handles.append(Line2D([0], [0], color="blue", marker="x", linestyle="None", markersize=8, label="policy"))
            handles.append(Line2D([0], [0], color="green", marker="x", linestyle="None", markersize=8, label="policy+cbf"))

        # Legend and labels
        ax.legend(handles=handles, loc="best", fontsize=fontsize - 2)
        ax.set_xlabel(metric_labels[metric_x], fontsize=fontsize)
        ax.set_ylabel(metric_labels[metric_y], fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)

        # Colorbar for tau
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([])
        cbar.set_label('tau', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

    # Turn off unused subplots
    for ax in axes[n_subplots:]:
        ax.axis("off")

    if title:
        ax.set_title(title, fontsize=fontsize)

    plt.tight_layout()
    return fig


pretty_names = {
    # scalar per-episode
    "cbf_activations_tot": "Total CBF activations",
    "cbf_when_uf": "CBF activations when UF active",
    "inner_steps_mean": "Mean inner steps",
    "mean_u_e": "Mean uncertainty estimate",
    "std_u_e": "Std uncertainty estimate",
    "steps": "Episode length",
    "total_collisions": "Total collisions",
    "total_length": "Total path length",
    "total_reward": "Total reward",
    "total_success": "Success (0/1)",
    "uf_activations_tot": "Total UF activations",
    "uf_when_cbf": "UF activations when CBF active",

    # list per-step
    "angle_ema": "Angle to EMA",
    "angle_goal": "Angle to goal",
    "cbf_activation_avg": "CBF activation (avg)",
    "cbf_mean_change": "CBF mean change",
    "dist_ema": "Distance to EMA",
    "dist_goal": "Distance to goal",
    "f_action": "Forward action",
    "f_velocity": "Forward velocity",
    "l_velocity": "Left velocity",
    "r_action": "Rotation action",
    "r_velocity": "Rotation velocity",
    "u_e": "Uncertainty estimate",
    "uf_activation": "UF activation",

    # rays
    "ray_0": "Ray 0",
    "ray_1": "Ray 1",
    "ray_2": "Ray 2",
    "ray_3": "Ray 3",
    "ray_4": "Ray 4",
    "ray_5": "Ray 5",
    "ray_6": "Ray 6",
    "ray_7": "Ray 7",
    "ray_8": "Ray 8",
    "ray_9": "Ray 9",
    "ray_10": "Ray 10",
    "ray_11": "Ray 11",
    "ray_12": "Ray 12",
    "ray_13": "Ray 13",
    "ray_14": "Ray 14",
    "ray_15": "Ray 15",
    "ray_16": "Ray 16",
    
    'ray_mean': 'Ray Average',
    'ray_std': 'Ray Standard Deviation'
}

def make_cmap_with_white(base_cmap='viridis', white_cut=0.05, cmap_range=(0.0, 1.0)):
    """
    Create a modified colormap with white replacing the lowest values.

    This function samples a base matplotlib colormap over a specified range 
    and replaces a fraction of the lowest values with pure white. It is useful 
    for visualizations where very low values should appear as background or 
    highlight regions.

    Parameters
    ----------
    base_cmap : str or Colormap, optional
        Name of the base colormap or a Colormap object (default is "viridis").
    white_cut : float, optional
        Fraction of the colormap (from the bottom) to replace with white, 
        between 0 and 1 (default is 0.05).
    cmap_range : tuple of float, optional
        Range of the base colormap to sample, as (min, max) values in [0, 1] 
        (default is (0.0, 1.0)).

    Returns
    -------
    matplotlib.colors.ListedColormap
        A modified colormap with the lowest values replaced by white.
    """

    # Get the base colormap
    base = plt.get_cmap(base_cmap)

    # Sample colors in the given range
    colors = base(np.linspace(cmap_range[0], cmap_range[1], 256))

    # Number of entries to replace with white
    n_white = int(256 * white_cut)
    if n_white > 0:
        colors[:n_white, :] = [1, 1, 1, 1]

    # Return modified colormap
    return ListedColormap(colors)

def plot_stats(
    all_stats,
    keys=None,
    filter_fn=lambda x: True,
    figsize=(6, 4),
    ax=None,
    bins=100,
    norm_x=True,
    norm_y=True,
    cmap='afmhot_r',
    white_cut=0.0,
    cmap_range=(0.0, 0.95),
    norm_type='log',
    gamma=0.5,
    v_line=None,
    h_line=None,
    outlier_quantile=0.02,
    fontsize=14
):
    """
    Plot statistical distributions or relationships from episode data.

    Depending on the `keys` argument, this function can:
      - Print available keys if `keys=None`
      - Plot a 2D histogram (heatmap) if `keys` is a tuple/list of two keys
      - Provide customization options such as outlier removal, normalization, 
        colormap adjustments, and reference lines.

    Parameters
    ----------
    all_stats : list of dict
        List of episode statistics dictionaries, typically produced by `extract_stats`.
    keys : tuple of str or None, optional
        Keys to plot. If None, available keys are printed. 
        If two keys are provided, a 2D histogram is plotted.
    filter_fn : callable, optional
        Function to filter episodes before plotting (default is `lambda x: True`).
    figsize : tuple of int, optional
        Figure size for matplotlib (default is (6, 4)).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    bins : int, optional
        Number of bins for histogram (default is 100).
    norm_x : bool, optional
        If True, normalize histogram counts along the x-axis (rows) (default is True).
    norm_y : bool, optional
        If True, normalize histogram counts along the y-axis (columns) (default is True).
    cmap : str, optional
        Base colormap name (default is "afmhot_r").
    white_cut : float, optional
        Fraction of the colormap replaced with white at the bottom (default is 0.0).
    cmap_range : tuple of float, optional
        Range of the base colormap to use (default is (0.0, 0.95)).
    norm_type : str or None, optional
        Normalization type for color scale: "log", "power", or None (default is "log").
    gamma : float, optional
        Gamma value for power normalization, only if `norm_type="power"` (default is 0.5).
    v_line : float, optional
        Vertical reference line at this x-value (default is None).
    h_line : float, optional
        Horizontal reference line at this y-value (default is None).
    outlier_quantile : float, optional
        Quantile for outlier removal (default is 0.02).
    fontsize : int, optional
        Font size for axis labels and ticks (default is 14).

    Returns
    -------
    matplotlib.figure.Figure or matplotlib.axes.Axes or None
        If a new figure is created, returns the matplotlib Figure. 
        If plotting on an existing axis, returns the Axes. 
        If `keys=None`, prints available keys and returns None.
    """

    filtered = [ep for ep in all_stats if filter_fn(ep)]
    n = len(filtered)
    if n == 0:
        print("No episodes matching filter.")
        return

    # Extract time series lists for a given key
    def get_time_series(key):
        return [ep[key] for ep in filtered if key in ep and isinstance(ep[key], list)]

    # Remove outliers based on quantiles
    def remove_outliers(x, y=None, q=0.01):
        if y is None:
            low, high = np.quantile(x, [q, 1 - q])
            return x[(x >= low) & (x <= high)]
        else:
            low_x, high_x = np.quantile(x, [q, 1 - q])
            low_y, high_y = np.quantile(y, [q, 1 - q])
            mask = (x >= low_x) & (x <= high_x) & (y >= low_y) & (y <= high_y)
            return x[mask], y[mask]

    # Help mode: print available keys
    if keys is None:
        scalar_keys = set()
        list_keys = set()
        for ep in filtered:
            for k, v in ep.items():
                if isinstance(v, list):
                    list_keys.add(k)
                elif isinstance(v, (int, float, np.number)):
                    scalar_keys.add(k)

        print(f"\nAvailable keys (from {n} episodes):\n")
        print("• Scalar per-episode:", ", ".join(sorted(scalar_keys)) or "None")
        print("• List per-step:", ", ".join(sorted(list_keys)) or "None")
        print("\nUsage:")
        print("  keys=('reward', 'time')   → time series")
        print("  keys=('score','duration') → 2D histogram")
        print("Extra options:")
        print("  norm_x=True / norm_y=True → normalize along axis")
        print("  cmap='plasma', white_cut=0.05, cmap_range=(0,0.9)")
        print("  norm_type='log' / 'power'")
        print("  outlier_quantile=0.01")
        print("  v_line=0.0, h_line=0.0")
        return

    # 2D histogram mode
    if isinstance(keys, (tuple, list)) and len(keys) == 2:
        key1, key2 = keys
        x_vals, y_vals = [], []

        # Handle special cases for ray sensors
        def get_value(ep, key):
            if key == "ray_mean":
                ray_keys = [f"ray_{i}" for i in range(17)]
                ray_lists = [ep.get(rk, []) for rk in ray_keys if isinstance(ep.get(rk, []), list)]
                if not ray_lists:
                    return None
                return np.mean(np.vstack(ray_lists), axis=0).tolist()
            if key == "ray_std":
                ray_keys = [f"ray_{i}" for i in range(17)]
                ray_lists = [ep.get(rk, []) for rk in ray_keys if isinstance(ep.get(rk, []), list)]
                if not ray_lists:
                    return None
                return np.std(np.vstack(ray_lists), axis=0).tolist()
            return ep.get(key, None)

        # Collect paired values
        for ep in filtered:
            v1, v2 = get_value(ep, key1), get_value(ep, key2)
            if v1 is None or v2 is None:
                continue
            if isinstance(v1, list) and isinstance(v2, list):
                l = min(len(v1), len(v2))
                x_vals.extend(v1[:l])
                y_vals.extend(v2[:l])
            elif not isinstance(v1, list) and not isinstance(v2, list):
                x_vals.append(v1)
                y_vals.append(v2)

        if not x_vals:
            print(f"No paired data for {key1} and {key2}")
            return

        x_vals, y_vals = np.array(x_vals), np.array(y_vals)

        # Apply outlier removal if needed
        if outlier_quantile is not None:
            x_vals, y_vals = remove_outliers(x_vals, y_vals, q=outlier_quantile)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        counts, xedges, yedges = np.histogram2d(x_vals, y_vals, bins=bins)

        # Normalize counts by rows or columns
        if norm_y:
            col_sums = counts.sum(axis=0, keepdims=True)
            col_sums[col_sums == 0] = 1
            counts = counts / col_sums
        if norm_x:
            row_sums = counts.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            counts = counts / row_sums

        # Build custom colormap
        cmap_used = make_cmap_with_white(cmap, white_cut=white_cut, cmap_range=cmap_range)
        X, Y = np.meshgrid(xedges, yedges)

        # Select normalization for color scale
        if norm_type == "log":
            norm = LogNorm(vmin=max(1e-6, counts[counts > 0].min()), vmax=counts.max())
        elif norm_type == "power":
            norm = PowerNorm(gamma, vmin=0, vmax=counts.max())
        else:
            norm = None

        # Plot heatmap
        im = ax.pcolormesh(X, Y, counts.T, cmap=cmap_used, shading='auto', norm=norm)
        cbar = ax.figure.colorbar(im, ax=ax, label='Counts')
        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label('Counts', fontsize=fontsize)

        # Axis labels and ticks
        ax.set_xlabel(pretty_names[key1], fontsize=fontsize)
        ax.set_ylabel(pretty_names[key2], fontsize=fontsize)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)

        # Reference guide lines
        if v_line is not None:
            ax.axvline(v_line, color="blue", linestyle="--", linewidth=2)
        if h_line is not None:
            ax.axhline(h_line, color="blue", linestyle=":", linewidth=2)

        if ax is None:
            return fig
        else:
            return ax

    print("Invalid input for keys parameter.")


####################################################################################################
####################################################################################################

#   ╔═════════════════════════════════════╗
#   ║   Model Score and Tabular Results   ║
#   ╚═════════════════════════════════════╝


def compute_score(df):
    """
    Compute a synthetic performance score from evaluation metrics.

    This function normalizes key metrics, inverts those where lower values 
    are better (episode length and collisions), and combines them with 
    success rate and velocity into a single score using the geometric mean.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing at least the following columns:
            - success_rate : float
            - forward_velocity_mean : float
            - length_mean : float
            - collisions_mean : float

    Returns
    -------
    pandas.DataFrame
        The input DataFrame with an additional column:
            - score : float
                Synthetic performance score in [0, 1].
    """

    # Normalize selected metrics to [0,1]
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[["forward_velocity_mean", "length_mean", "collisions_mean"]] = scaler.fit_transform(
        df[["forward_velocity_mean", "length_mean", "collisions_mean"]]
    )

    # Invert length and collisions (lower is better)
    df_norm["length_score"] = 1 - df_norm["length_mean"]
    df_norm["collision_score"] = 1 - df_norm["collisions_mean"]

    # Select positive metrics
    metrics = [
        df["success_rate"],                # already in [0,1]
        df_norm["forward_velocity_mean"],  # normalized
        df_norm["length_score"],           # inverted and normalized
        df_norm["collision_score"]         # inverted and normalized
    ]

    # Geometric mean to combine metrics
    df["score"] = (metrics[0] * metrics[1] * metrics[2] * metrics[3]) ** (1/4)
    return df


def analyze_folder(folder_path):
    """
    Analyze experiment results in a folder and compute performance summaries.

    This function processes all result CSV files ending with "_all.csv" in a given
    folder, computes synthetic scores via `compute_score`, and generates multiple
    summary DataFrames. These summaries highlight the best-performing uncertainty
    estimation models across environments and percentiles.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing experiment CSV result files.

    Returns
    -------
    tuple of pandas.DataFrame
        - summary_global : DataFrame
            Best percentile configuration for each model across all environments,
            sorted by score.
        - summary_by_env : DataFrame
            Performance of the globally best percentiles broken down by environment.
        - summary_best_per_env : DataFrame
            Best percentile configuration selected independently for each model
            and environment.
        - summary_st_as_ref : DataFrame
            Performance results when applying the best percentile chosen on the ST
            environment across all environments, including aggregate groups (ALL, OOD).

    Notes
    -----
    Columns are standardized and reordered for readability:
        - model : str
        - env. : str (if applicable)
        - $\\tau$ : int (percentile threshold)
        - score : float
        - Success rate : float
        - Collisions : float
        - Velocity : float
        - Ep. length : float
    """

    all_rows = []

    for file in os.listdir(folder_path):
        if file.endswith("_all.csv"):
            filepath = os.path.join(folder_path, file)
            env, model, _ = file.split("_", 2)  # extract environment and model
            
            df = pd.read_csv(filepath)
            df = compute_score(df)
            df["environment"] = env
            df["model"] = model + ' UE'

            # Special case: baseline model
            if model == "base":
                for cfg, name in [(0, "policy"), (1, "policy+cbf")]:
                    df_cfg = df[df["cbf_config"] == cfg].copy()
                    df_cfg["model"] = name
                    df_cfg["percentile"] = -1
                    all_rows.append(df_cfg)
            else:
                all_rows.append(df)

    # Concatenate all rows
    all_data = pd.concat(all_rows, ignore_index=True)

    # Global analysis: best percentile per model
    grouped = all_data.groupby(["model", "percentile"]).agg({
        "score": "mean",
        "success_rate": "mean",
        "collisions_mean": "mean",
        "forward_velocity_mean": "mean",
        "length_mean": "mean"
    }).reset_index()
    best_perc = grouped.loc[grouped.groupby("model")["score"].idxmax()]
    summary_global = best_perc.copy().sort_values("score", ascending=False).reset_index(drop=True)

    # Per-environment performance of global best percentiles
    summary_by_env = []
    for _, row in best_perc.iterrows():
        model = row["model"]
        perc = row["percentile"]
        df_model = all_data[(all_data["model"] == model) & (all_data["percentile"] == perc)]
        agg = df_model.groupby(["model", "percentile", "environment"]).agg({
            "score": "mean",
            "success_rate": "mean",
            "collisions_mean": "mean",
            "forward_velocity_mean": "mean",
            "length_mean": "mean"
        }).reset_index()
        summary_by_env.append(agg)
    summary_by_env = pd.concat(summary_by_env, ignore_index=True)

    # Best percentile per model and environment
    grouped_env = all_data.groupby(["environment", "model", "percentile"]).agg({
        "score": "mean",
        "success_rate": "mean",
        "collisions_mean": "mean",
        "forward_velocity_mean": "mean",
        "length_mean": "mean"
    }).reset_index()
    best_perc_env = grouped_env.loc[grouped_env.groupby(["environment", "model"])["score"].idxmax()]
    summary_best_per_env = best_perc_env.copy()

    # ST as reference: apply best ST percentile across environments
    summary_st_as_ref = []
    st_rows = best_perc_env[best_perc_env["environment"] == "ST"]

    for _, row in st_rows.iterrows():
        model = row["model"]
        perc = row["percentile"]
        df_model = all_data[(all_data["model"] == model) & (all_data["percentile"] == perc)]
        agg = df_model.groupby(["model", "percentile", "environment"]).agg({
            "score": "mean",
            "success_rate": "mean",
            "collisions_mean": "mean",
            "forward_velocity_mean": "mean",
            "length_mean": "mean"
        }).reset_index()

        # Add ALL and OOD aggregate groups
        agg_all = agg.groupby(["model", "percentile"]).mean(numeric_only=True).reset_index()
        agg_all["environment"] = "ALL"

        agg_ood = agg[agg["environment"].isin(["SCW", "MO"])].groupby(["model", "percentile"]).mean(numeric_only=True).reset_index()
        agg_ood["environment"] = "OOD"

        agg = pd.concat([agg, agg_all, agg_ood], ignore_index=True)
        summary_st_as_ref.append(agg)

    summary_st_as_ref = pd.concat(summary_st_as_ref, ignore_index=True)

    # Column renaming and reordering
    def rename_and_reorder(df, include_env=True):
        mapping = {
            "environment": "env.",
            "percentile": "$\\tau$",
            "success_rate": "Success rate",
            "collisions_mean": "Collisions",
            "forward_velocity_mean": "Velocity",
            "length_mean": "Ep. length"
        }
        df = df.rename(columns=mapping)

        # Standardize model names
        model_mapping = {
            "mcd UE": "MCD UE",
            "rnd UE": "RND UE",
            "qnet UE": "Ens. UE",
            "prob UE": "Prob. UE",
            "random UE": "Random UE",
            "policy": "Policy",
            "policy+cbf": "Policy + CBF"
        }
        if "model" in df.columns:
            df["model"] = df["model"].replace(model_mapping)

        cols = ["model"]
        if include_env and "env." in df.columns:
            cols.append("env.")
        cols += ["$\\tau$", "score", "Success rate", "Collisions", "Velocity", "Ep. length"]

        return df[cols]

    summary_global = rename_and_reorder(summary_global)
    summary_by_env = rename_and_reorder(summary_by_env)
    summary_best_per_env = rename_and_reorder(summary_best_per_env)
    summary_st_as_ref = rename_and_reorder(summary_st_as_ref)

    return summary_global, summary_by_env, summary_best_per_env, summary_st_as_ref
