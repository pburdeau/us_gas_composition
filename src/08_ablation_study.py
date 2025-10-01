#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare methods (Our method, Ordinary kriging, Nearest neighbor) across distance and variance bins,
optionally propagate normalization uncertainties, and generate figures. Also plots std-comparison
histograms for kriging vs. non-linear model uncertainty proxies.
"""

# =========================
# Imports & Global Config
# =========================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---- CONFIG: update if needed ----
root_path = '/scratch/users/pburdeau/data/gas_composition'
out_path = os.path.join(root_path, 'out')
os.makedirs(out_path, exist_ok=True)


# =========================================
# Uncertainty propagation for normalization
# =========================================
def propagate_uncertainties_normalization(data_on_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Re-normalize component columns (with '_y' suffix) to sum to 100 and propagate uncertainties.

    Expects:
      - value columns: ['HE_y', 'CO2_y', ..., 'C6+_y']
      - std columns (before): ['std_HE', 'std_CO2', ..., 'std_C6+']
      - baseline columns for errors: '{base}_x' and 'std_C1' (for nse)

    Returns:
      data_on_grid with updated normalized values, propagated stds, and error columns.
    """
    components_base = ['HE','CO2','H2','N2','H2S','AR','O2','C1','C2','C3','N-C4','I-C4','N-C5','I-C5','C6+']
    components = [f'{c}_y' for c in components_base]

    # Correlation/cov computed on available pairs; retained but not used directly below (kept as in your code)
    _ = data_on_grid[components].corr()
    _ = data_on_grid[components].cov()

    list_stds = [f'std_{c}' for c in components_base]

    values = data_on_grid[components].values.astype(float)
    uncertainties = data_on_grid[list_stds].values.astype(float)

    # Sum & uncertainty; guard zeros
    sum_values = np.nansum(values, axis=1)
    sum_values_safe = np.where(sum_values == 0, np.nan, sum_values)

    # (Conservative) combine stds in quadrature
    uncertainty_sum = np.sqrt(np.nansum(uncertainties**2, axis=1))

    # Normalize to sum 100
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = values * (100.0 / sum_values_safe[:, None])

    for i, comp in enumerate(components):
        data_on_grid[comp] = normalized[:, i]

    # Fractional uncertainty for each component; guard zeros in value
    vals_safe = np.where(values == 0, np.nan, values)
    frac_unc = np.sqrt((uncertainties / vals_safe)**2 + (uncertainty_sum[:, None] / sum_values_safe[:, None])**2)

    # Absolute uncertainty (propagate to normalized values)
    abs_unc = frac_unc * normalized / 100.0 * sum_values_safe[:, None]  # keep proportional to normalized value

    # Write back per-component stds
    for j, base in enumerate(components_base):
        data_on_grid[f'std_{base}'] = abs_unc[:, j]

    # Sum check
    data_on_grid['sum_check'] = data_on_grid[components].sum(axis=1).round(2)

    # Recompute error metrics for C1 (matches your original)
    comp_y = 'C1_y'
    base = 'C1'
    if f'{base}_x' in data_on_grid.columns and 'std_C1' in data_on_grid.columns:
        data_on_grid['abs_error'] = np.abs(data_on_grid[comp_y] - data_on_grid[f'{base}_x'])
        data_on_grid['sq_error'] = (data_on_grid[comp_y] - data_on_grid[f'{base}_x'])**2
        with np.errstate(divide='ignore', invalid='ignore'):
            data_on_grid['nse'] = ((data_on_grid[comp_y] - data_on_grid[f'{base}_x']) / data_on_grid['std_C1'])**2

    return data_on_grid


# =========================================
# Main comparison plotting (distance/std)
# =========================================
def plot_comparison_lists_v4(
    data_source: str,
    nbins: int,
    threshold: float,
    comp: str,
    merged_simple_kriging_list: list[pd.DataFrame],
    merged_kriging_list: list[pd.DataFrame],
    merged_nn_list: list[pd.DataFrame] | None = None,
    propagate: bool = False,
    select_basin: bool = False,
    basin_list: list[str] | None = None,
    change_beta: bool = False,
    new_beta: float | None = None,
):
    """
    Compare methods across distance and variance bins with optional normalization propagation.
    Saves EPS figures to the CWD.
    """
    # Optional basin filtering
    if select_basin and basin_list is not None:
        merged_simple_kriging_list = [
            df[df.BASIN_NAME.isin(basin_list)] for df in merged_simple_kriging_list
        ]
        merged_kriging_list = [
            df[df.BASIN_NAME.isin(basin_list)] for df in merged_kriging_list
        ]
        if merged_nn_list is not None:
            merged_nn_list = [df[df.BASIN_NAME.isin(basin_list)] for df in merged_nn_list]

    # Local beta combiner
    def predic_beta(pred1, pred_1_std, pred2, beta):
        w_krig = 1.0
        w_pred = beta * (pred_1_std**2)
        return (pred1 + w_pred * pred2) / (w_krig + w_pred)

    # If change_beta: recompute y from kriging+NN using the new beta, keep errors consistent
    if change_beta and new_beta is not None:
        merged_kriging_list_new = []
        for df_k, df_simple in zip(merged_kriging_list, merged_simple_kriging_list):
            df = df_k.copy()
            df_simple = df_simple.copy()
            df[f'{comp}_y'] = predic_beta(
                df_simple[f'{comp}_y'],
                df[f'std_{comp}'],
                df[f'{comp}_predic_gor'],
                new_beta
            )
            df['abs_error'] = np.abs(df[f'{comp}_y'] - df[f'{comp}_x'])
            df['sq_error'] = (df[f'{comp}_y'] - df[f'{comp}_x'])**2
            merged_kriging_list_new.append(df)
        merged_kriging_list = merged_kriging_list_new

    # Quick counts before/after NaN filtering
    print('simple_kriging', len(pd.concat(merged_simple_kriging_list, ignore_index=True)))
    print('kriging', len(pd.concat(merged_kriging_list, ignore_index=True)))
    if merged_nn_list is not None:
        print('nn', len(pd.concat(merged_nn_list, ignore_index=True)))

    # Require all core columns present, drop NaNs, drop duplicates
    needed_cols = ['abs_error', 'sq_error', 'closest_distance', f'std_{comp}']
    merged_simple_kriging_list = [
        d.dropna(subset=needed_cols).drop_duplicates() for d in merged_simple_kriging_list
    ]
    merged_kriging_list = [
        d.dropna(subset=needed_cols).drop_duplicates() for d in merged_kriging_list
    ]
    if merged_nn_list is not None:
        merged_nn_list = [
            d.dropna(subset=needed_cols).drop_duplicates() for d in merged_nn_list
        ]

    print('simple_kriging', len(pd.concat(merged_simple_kriging_list, ignore_index=True)))
    print('kriging', len(pd.concat(merged_kriging_list, ignore_index=True)))
    if merged_nn_list is not None:
        print('nn', len(pd.concat(merged_nn_list, ignore_index=True)))

    # Assemble dicts
    data_dict = {
        'Our method': merged_kriging_list,
        'Ordinary kriging': merged_simple_kriging_list,
        'Nearest neighbor': merged_nn_list
    }
    data_dict = {k: v for k, v in data_dict.items() if v is not None}  # drop None

    data_dict_var = {  # only methods with kriging std
        'Our method': merged_kriging_list,
        'Ordinary kriging': merged_simple_kriging_list
    }

    # Equal-size binning helper
    def select_equal_size_bins(df_list, column, n_bins=nbins, threshold=threshold):
        concatenated_df = pd.concat(df_list, ignore_index=True)
        if propagate:
            concatenated_df = propagate_uncertainties_normalization(concatenated_df)

        series = concatenated_df[column].dropna().sort_values()
        if len(series) == 0:
            # fallback single bin
            return [0.0, 1.0]

        # keep lower 'threshold' quantile (e.g., 0.5, 0.8); remove only highest tail
        cutoff = np.quantile(series, threshold)
        series = series[series <= cutoff]

        # if too few points, just two edges (min, max) to avoid pd.cut errors
        if len(series) < 3 or n_bins < 2:
            return [float(series.iloc[0]), float(series.iloc[-1])]

        # equal-count bin edges
        edges = [float(series.iloc[0])]
        for i in range(1, n_bins):
            idx = int(len(series) * i / n_bins)
            edges.append(float(series.iloc[idx]))
        edges.append(float(series.iloc[-1]))
        edges = sorted(set(edges))
        if len(edges) < 2:
            edges = [float(series.min()), float(series.max())]
        return edges

    # Build bin edges
    distance_bin_edges = select_equal_size_bins(
        [df for dfs in data_dict.values() for df in dfs],
        'closest_distance'
    )
    variance_bin_edges = select_equal_size_bins(
        [df for dfs in data_dict_var.values() for df in dfs],
        f'std_{comp}'
    )

    # Stats helper
    def calculate_stats(df_list, bin_column, bins):
        concatenated_df = pd.concat(df_list, ignore_index=True)
        if propagate:
            concatenated_df = propagate_uncertainties_normalization(concatenated_df)

        # pd.cut requires at least 2 unique edges
        if len(bins) < 2 or np.all(np.array(bins) == bins[0]):
            # create trivial bins around the data
            col = concatenated_df[bin_column].dropna()
            if col.empty:
                concatenated_df['bin'] = pd.Categorical([])
            else:
                bmin, bmax = float(col.min()), float(col.max())
                bins = [bmin, bmax if bmax > bmin else bmin + 1e-6]
                concatenated_df['bin'] = pd.cut(concatenated_df[bin_column], bins=bins, include_lowest=True)
        else:
            concatenated_df['bin'] = pd.cut(concatenated_df[bin_column], bins=bins, include_lowest=True)

        mae = concatenated_df.groupby('bin')['abs_error'].mean()
        rmse = np.sqrt(concatenated_df.groupby('bin')['sq_error'].mean())
        counts = concatenated_df['bin'].value_counts().sort_index()
        mae_se = concatenated_df.groupby('bin')['abs_error'].sem()
        rmse_se = concatenated_df.groupby('bin').apply(lambda x: np.sqrt(x['sq_error']).sem())
        return mae, rmse, counts, mae_se, rmse_se, concatenated_df

    # Compute statistics
    stats_distance = {name: calculate_stats(dfs, 'closest_distance', distance_bin_edges) for name, dfs in data_dict.items()}
    stats_variance = {name: calculate_stats(dfs, f'std_{comp}', variance_bin_edges) for name, dfs in data_dict_var.items()}

    # Distance-binned frames
    df_mae_distance = pd.DataFrame({name: stat[0] for name, stat in stats_distance.items()})
    df_rmse_distance = pd.DataFrame({name: stat[1] for name, stat in stats_distance.items()})
    df_histogram_distance = pd.DataFrame({name: stat[2] for name, stat in stats_distance.items()})
    df_mae_se_distance = pd.DataFrame({name: stat[3] for name, stat in stats_distance.items()})
    df_rmse_se_distance = pd.DataFrame({name: stat[4] for name, stat in stats_distance.items()})

    # For scatter or debug
    concatenated_kriging_df = stats_distance['Our method'][5]
    concatenated_simple_kriging_df = stats_distance['Ordinary kriging'][5]

    # Variance-binned frames
    df_mae_variance = pd.DataFrame({name: stat[0] for name, stat in stats_variance.items()})
    df_rmse_variance = pd.DataFrame({name: stat[1] for name, stat in stats_variance.items()})
    df_histogram_variance = pd.DataFrame({name: stat[2] for name, stat in stats_variance.items()})
    df_mae_se_variance = pd.DataFrame({name: stat[3] for name, stat in stats_variance.items()})
    df_rmse_se_variance = pd.DataFrame({name: stat[4] for name, stat in stats_variance.items()})

    # Fill NAs with zeros for plotting
    for df_ in [df_mae_distance, df_rmse_distance, df_histogram_distance, df_mae_se_distance, df_rmse_se_distance,
                df_mae_variance, df_rmse_variance, df_histogram_variance, df_mae_se_variance, df_rmse_se_variance]:
        df_.fillna(0, inplace=True)

    # Pretty bin labels
    df_mae_distance.index = df_mae_distance.index.map(lambda x: f"{int(x.left/1000)} - {int(x.right/1000)} km")
    df_rmse_distance.index = df_rmse_distance.index.map(lambda x: f"{int(x.left/1000)} - {int(x.right/1000)} km")
    df_histogram_distance.index = df_histogram_distance.index.map(lambda x: f"{int(x.left/1000)} - {int(x.right/1000)} km")
    df_mae_se_distance.index = df_mae_se_distance.index.map(lambda x: f"{int(x.left/1000)} - {int(x.right/1000)} km")
    df_rmse_se_distance.index = df_rmse_se_distance.index.map(lambda x: f"{int(x.left/1000)} - {int(x.right/1000)} km")

    df_mae_variance.index = df_mae_variance.index.map(lambda x: f"{x.left:.0f} - {x.right:.0f} %")
    df_rmse_variance.index = df_rmse_variance.index.map(lambda x: f"{x.left:.0f} - {x.right:.0f} %")
    df_histogram_variance.index = df_histogram_variance.index.map(lambda x: f"{x.left:.0f} - {x.right:.0f} %")
    df_mae_se_variance.index = df_mae_se_variance.index.map(lambda x: f"{x.left:.0f} - {x.right:.0f} %")
    df_rmse_se_variance.index = df_rmse_se_variance.index.map(lambda x: f"{x.left:.0f} - {x.right:.0f} %")

    # Relative improvements (distance)
    df_diff_mae_pct = (df_mae_distance['Our method'] - df_mae_distance['Ordinary kriging']) / df_mae_distance['Ordinary kriging'] * 100.0
    df_diff_rmse_pct = (df_rmse_distance['Our method'] - df_rmse_distance['Ordinary kriging']) / df_rmse_distance['Ordinary kriging'] * 100.0
    mae_diff_se = np.sqrt(df_mae_se_distance['Our method']**2 + df_mae_se_distance['Ordinary kriging']**2)
    rmse_diff_se = np.sqrt(df_rmse_se_distance['Our method']**2 + df_rmse_se_distance['Ordinary kriging']**2)
    # SE propagated to % difference
    diff_mae_pct_se = np.sqrt(
        (100.0 / df_mae_distance['Ordinary kriging'])**2 * df_mae_se_distance['Our method']**2 +
        (-100.0 * (df_mae_distance['Our method'] - df_mae_distance['Ordinary kriging']) / df_mae_distance['Ordinary kriging']**2)**2 * df_mae_se_distance['Ordinary kriging']**2
    )
    diff_rmse_pct_se = np.sqrt(
        (100.0 / df_rmse_distance['Ordinary kriging'])**2 * df_rmse_se_distance['Our method']**2 +
        (-100.0 * (df_rmse_distance['Our method'] - df_rmse_distance['Ordinary kriging']) / df_rmse_distance['Ordinary kriging']**2)**2 * df_rmse_se_distance['Ordinary kriging']**2
    )

    # Relative improvements (variance)
    df_diff_mae_pct_var = (df_mae_variance['Our method'] - df_mae_variance['Ordinary kriging']) / df_mae_variance['Ordinary kriging'] * 100.0
    df_diff_rmse_pct_var = (df_rmse_variance['Our method'] - df_rmse_variance['Ordinary kriging']) / df_rmse_variance['Ordinary kriging'] * 100.0
    mae_diff_se_var = np.sqrt(df_mae_se_variance['Our method']**2 + df_mae_se_variance['Ordinary kriging']**2)
    rmse_diff_se_var = np.sqrt(df_rmse_se_variance['Our method']**2 + df_rmse_se_variance['Ordinary kriging']**2)
    diff_mae_pct_se_var = np.sqrt(
        (100.0 / df_mae_variance['Ordinary kriging'])**2 * df_mae_se_variance['Our method']**2 +
        (-100.0 * (df_mae_variance['Our method'] - df_mae_variance['Ordinary kriging']) / df_mae_variance['Ordinary kriging']**2)**2 * df_mae_se_variance['Ordinary kriging']**2
    )
    diff_rmse_pct_se_var = np.sqrt(
        (100.0 / df_rmse_variance['Ordinary kriging'])**2 * df_rmse_se_variance['Our method']**2 +
        (-100.0 * (df_rmse_variance['Our method'] - df_rmse_variance['Ordinary kriging']) / df_rmse_variance['Ordinary kriging']**2)**2 * df_rmse_se_variance['Ordinary kriging']**2
    )

    # Color definitions
    method_colors = {
        'Our method': '#6d7a78',
        'Ordinary kriging': '#a7b0b0',
        'Nearest neighbor': '#8b9aaf'
    }
    improvement_colors = {
        'MAE Improvement': '#718882',
        'RMSE Improvement': '#c4c3c6'
    }
    S = 28

    # ----- Absolute MAE distance -----
    fig, ax = plt.subplots(figsize=(12, 8))
    df_mae_distance.plot(kind='bar', ax=ax, width=0.5, yerr=df_mae_se_distance, capsize=4,
                         color=[method_colors[col] for col in df_mae_distance.columns])
    ax.set_xlabel('', fontsize=S)
    ax.set_ylabel('Average MAE [mol %]', fontsize=S)
    ax.legend(data_dict.keys(), fontsize=S, loc='lower right')
    ax.set_xticklabels(df_mae_distance.index, rotation=45, ha='right', fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{data_source}_absolute_MAE_{comp}_distance.eps', format='eps', dpi=300)
    plt.show()

    # ----- Absolute RMSE distance -----
    fig, ax = plt.subplots(figsize=(12, 8))
    df_rmse_distance.plot(kind='bar', ax=ax, width=0.5, yerr=df_rmse_se_distance, capsize=4,
                          color=[method_colors[col] for col in df_rmse_distance.columns])
    ax.set_xlabel('', fontsize=S)
    ax.set_ylabel('Average RMSE [mol %]', fontsize=S)
    ax.legend(data_dict.keys(), fontsize=S, loc='lower right')
    ax.set_xticklabels(df_rmse_distance.index, rotation=45, ha='right', fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{data_source}_absolute_RMSE_{comp}_distance.eps', format='eps', dpi=300)
    plt.show()

    # Relative improvements (distance)
    bin_labels = df_diff_mae_pct.index
    color_histogram = improvement_colors["MAE Improvement"]
    color_shaded_nn = improvement_colors["RMSE Improvement"]

    mean_mae_improvement = -df_diff_mae_pct
    mean_rmse_improvement = -df_diff_rmse_pct
    lower_mae = mean_mae_improvement - diff_mae_pct_se
    upper_mae = mean_mae_improvement + diff_mae_pct_se
    lower_rmse = mean_rmse_improvement - diff_rmse_pct_se
    upper_rmse = mean_rmse_improvement + diff_rmse_pct_se

    # ----- Relative MAE distance -----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bin_labels, mean_mae_improvement, label="Relative improvement in MAE\nfrom ordinary kriging to our method",
            color=color_histogram, marker="o", linestyle="-")
    ax.fill_between(bin_labels, lower_mae, upper_mae, color=color_histogram, alpha=0.1)
    ax.set_xlabel("", fontsize=S)
    ax.set_ylabel("", fontsize=S)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.legend(fontsize=S, loc="lower right")
    ax.grid(True, linestyle="dotted", alpha=0.6)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.tight_layout()
    plt.savefig(f"{data_source}_relative_MAE_{comp}_distance.eps", format="eps", dpi=300)
    plt.show()

    # ----- Relative RMSE distance -----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bin_labels, mean_rmse_improvement, label="Relative improvement in RMSE\nfrom ordinary kriging to our method",
            color=color_shaded_nn, marker="o", linestyle="-")
    ax.fill_between(bin_labels, lower_rmse, upper_rmse, color=color_shaded_nn, alpha=0.1)
    ax.set_xlabel("", fontsize=S)
    ax.set_ylabel("", fontsize=S)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.legend(fontsize=S, loc="lower right")
    ax.grid(True, linestyle="dotted", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{data_source}_relative_RMSE_{comp}_distance.eps", format="eps", dpi=300)
    plt.show()

    # ----- Absolute MAE variance -----
    fig, ax = plt.subplots(figsize=(12, 8))
    df_mae_variance.plot(kind='bar', ax=ax, width=0.5, yerr=df_mae_se_variance, capsize=4,
                         color=[method_colors[col] for col in df_mae_variance.columns])
    ax.set_xlabel('Kriging standard deviation bin', fontsize=S)
    ax.set_ylabel('Average MAE [mol %]', fontsize=S)
    ax.legend(data_dict.keys(), fontsize=S, loc='lower right')
    ax.set_xticklabels(df_mae_variance.index, rotation=45, ha='right', fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{data_source}_absolute_MAE_{comp}_std.eps', format="eps", dpi=300)
    plt.show()

    # ----- Absolute RMSE variance -----
    fig, ax = plt.subplots(figsize=(12, 8))
    df_rmse_variance.plot(kind='bar', ax=ax, width=0.5, yerr=df_rmse_se_variance, capsize=4,
                          color=[method_colors[col] for col in df_rmse_variance.columns])
    ax.set_xlabel('Kriging standard deviation bin', fontsize=S)
    ax.set_ylabel('Average RMSE [mol %]', fontsize=S)
    ax.legend(data_dict.keys(), fontsize=S, loc='lower right')
    ax.set_xticklabels(df_rmse_variance.index, rotation=45, ha='right', fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{data_source}_absolute_RMSE_{comp}_std.eps', format="eps", dpi=300)
    plt.show()

    # ----- Histogram distance -----
    fig, ax = plt.subplots(figsize=(12, 8))
    df_histogram_distance.plot(kind='bar', ax=ax, width=0.5,
                               color=[method_colors[col] for col in df_histogram_distance.columns])
    ax.set_xlabel('Distance bin', fontsize=S)
    ax.set_ylabel('Number of points', fontsize=S)
    ax.legend(data_dict.keys(), fontsize=S)
    ax.set_xticklabels(df_histogram_distance.index, rotation=45, ha='right', fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ----- Histogram variance -----
    fig, ax = plt.subplots(figsize=(12, 8))
    df_histogram_variance.plot(kind='bar', ax=ax, width=0.5,
                               color=[method_colors[col] for col in df_histogram_variance.columns])
    print(df_histogram_variance.iloc[0])
    ax.set_xlabel('Kriging Standard Deviation Bin', fontsize=S)
    ax.set_ylabel('Number of points', fontsize=S)
    ax.legend(data_dict_var.keys(), fontsize=S)
    ax.set_xticklabels(df_histogram_variance.index, rotation=45, ha='right', fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ----- Relative MAE variance -----
    bin_labels = df_diff_mae_pct_var.index
    color_histogram = improvement_colors["MAE Improvement"]
    color_shaded_nn = improvement_colors["RMSE Improvement"]
    mean_mae_improvement = -df_diff_mae_pct_var
    mean_rmse_improvement = -df_diff_rmse_pct_var
    lower_mae = mean_mae_improvement - diff_mae_pct_se_var
    upper_mae = mean_mae_improvement + diff_mae_pct_se_var
    lower_rmse = mean_rmse_improvement - diff_rmse_pct_se_var
    upper_rmse = mean_rmse_improvement + diff_rmse_pct_se_var

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bin_labels, mean_mae_improvement, label="Relative improvement in MAE\nfrom ordinary kriging to our method",
            color=color_histogram, marker="o", linestyle="-")
    ax.fill_between(bin_labels, lower_mae, upper_mae, color=color_histogram, alpha=0.1)
    ax.set_xlabel("", fontsize=S)
    ax.set_ylabel("", fontsize=S)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.legend(fontsize=S, loc="lower right")
    ax.grid(True, linestyle="dotted", alpha=0.6)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    plt.tight_layout()
    plt.savefig(f"{data_source}_relative_MAE_{comp}_std.eps", format="eps", dpi=300)
    plt.show()

    # ----- Relative RMSE variance -----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bin_labels, mean_rmse_improvement, label="Relative improvement in RMSE\nfrom ordinary kriging to our method",
            color=color_shaded_nn, marker="o", linestyle="-")
    ax.fill_between(bin_labels, lower_rmse, upper_rmse, color=color_shaded_nn, alpha=0.1)
    ax.set_xlabel("", fontsize=S)
    ax.set_ylabel("", fontsize=S)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=S)
    ax.tick_params(axis='y', labelsize=S)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    ax.legend(fontsize=S, loc="lower right")
    ax.grid(True, linestyle="dotted", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{data_source}_relative_RMSE_{comp}_std.eps", format="eps", dpi=300)
    plt.show()


# =========================================
# Distance-threshold line comparison (NN)
# =========================================
def run_distance_threshold(comp: str, list_suffix: str):
    """
    Plot line comparisons (Ordinary kriging vs Non-linear model) across equal-size distance bins.
    Uses pre-populated 'all_data' dict (see bottom of file).
    """
    simple_df_list = all_data[f'merged_simple_kriging_list_{comp}_{list_suffix}']
    kriging_df_list = all_data[f'merged_kriging_list_{comp}_{list_suffix}']
    nn_df_list = all_data[f'merged_nn_list_{comp}_{list_suffix}']

    # Compute abs_error of NN (non-linear) vs truth
    for df in kriging_df_list:
        df["abs_error_nn"] = np.abs(df[f"{comp}_x"] - df[f"{comp}_predic_gor"])

    def bin_and_compute_stats(df_list, bin_column, error_column, n_bins=25, threshold=1.0):
        concatenated_df = pd.concat(df_list, ignore_index=True)
        sorted_values = concatenated_df[bin_column].dropna().sort_values()

        # keep lower fraction (threshold=1 keeps all; use <1 to trim tail)
        upper_cutoff = np.quantile(sorted_values, threshold)
        filtered_values = sorted_values[sorted_values <= upper_cutoff]

        if len(filtered_values) < 3 or n_bins < 2:
            bin_edges = [float(filtered_values.min()), float(filtered_values.max())] if len(filtered_values) > 0 else [0.0, 1.0]
        else:
            bin_edges = [float(filtered_values.iloc[0])]
            for i in range(1, n_bins):
                idx = int(len(filtered_values) * i / n_bins)
                bin_edges.append(float(filtered_values.iloc[idx]))
            bin_edges.append(float(filtered_values.iloc[-1]))
            bin_edges = sorted(set(bin_edges))
            if len(bin_edges) < 2:
                bin_edges = [float(filtered_values.min()), float(filtered_values.max())]

        concatenated_df["bin"] = pd.cut(concatenated_df[bin_column], bins=bin_edges, include_lowest=True)

        mean_abs_error = concatenated_df.groupby("bin")[error_column].mean()
        se_abs_error = concatenated_df.groupby("bin")[error_column].sem()
        return mean_abs_error, se_abs_error, bin_edges

    mean_abs_error_A, se_abs_error_A, bin_edges = bin_and_compute_stats(kriging_df_list, "closest_distance", "abs_error")
    mean_abs_error_B, se_abs_error_B, _ = bin_and_compute_stats(kriging_df_list, "closest_distance", "abs_error_nn")

    bin_labels = [f"{int(interval.left / 1000)} - {int(interval.right / 1000)} km" for interval in mean_abs_error_A.index]
    lower_A, upper_A = mean_abs_error_A - se_abs_error_A, mean_abs_error_A + se_abs_error_A
    lower_B, upper_B = mean_abs_error_B - se_abs_error_B, mean_abs_error_B + se_abs_error_B

    color_histogram = "#6b756f"  # Ordinary kriging
    color_mean_nn = "#533a8b"     # Non-linear model mean
    color_shaded_nn = "#533a8b"   # Non-linear model shaded SE

    LARGE_FONT = 22
    MEDIUM_FONT = 18

    plt.figure(figsize=(10, 8))
    plt.plot(bin_labels, mean_abs_error_A, label="Ordinary kriging", color=color_histogram, marker="o", linestyle="-")
    plt.plot(bin_labels, mean_abs_error_B, label="Non-linear model", color=color_mean_nn, marker="o", linestyle="-")

    plt.fill_between(bin_labels, lower_A, upper_A, color=color_histogram, alpha=0.2)
    plt.fill_between(bin_labels, lower_B, upper_B, color=color_shaded_nn, alpha=0.2)

    plt.xlabel("", fontsize=LARGE_FONT)
    plt.ylabel("Mean Absolute Error", fontsize=LARGE_FONT)
    plt.xticks(fontsize=MEDIUM_FONT)
    plt.yticks(fontsize=MEDIUM_FONT)
    plt.xticks(ticks=range(len(bin_labels)), labels=bin_labels, rotation=45, ha="right", fontsize=LARGE_FONT)

    legend = plt.legend(fontsize=MEDIUM_FONT, title_fontsize=MEDIUM_FONT, loc="upper right", frameon=True)
    for text in legend.get_texts():
        text.set_ha("right")

    plt.grid(True, linestyle="dotted", alpha=0.6)
    plt.tight_layout()
    plt.show()


# =========================================
# Std distribution comparison figure
# =========================================
def run_comparison_std(data_source: str, comp: str, list_suffix: str):
    """
    Compare kriging std distribution against the aggregated non-linear model uncertainty proxy (1/sqrt(beta)).
    Saves PNG to the CWD.
    """
    simple_df_list = all_data[f'merged_simple_kriging_list_{comp}_{list_suffix}']
    kriging_df_list = all_data[f'merged_kriging_list_{comp}_{list_suffix}']
    nn_df_list = all_data[f'merged_nn_list_{comp}_{list_suffix}']

    std_vals = np.concatenate([df[f"std_{comp}"].dropna().values for df in kriging_df_list]) if kriging_df_list else np.array([])

    if std_vals.size == 0:
        print("No std values available to plot.")
        return

    lower_std, upper_std = np.percentile(std_vals, [2.5, 97.5])
    mean_std = np.mean(std_vals)

    # Compute 1/sqrt(beta_{comp}) means across the four (a,b) splits (only if positive)
    beta_means = []
    for df in kriging_df_list:
        beta_col = f"beta_{comp}"
        if beta_col in df.columns:
            vals = df[beta_col].dropna().values
            vals = vals[vals > 0]
            if vals.size > 0:
                beta_means.append(np.mean(1.0 / np.sqrt(vals)))
    if not beta_means:
        print("No positive beta values found; skipping non-linear model uncertainty overlay.")
        beta_means = [np.nan, np.nan, np.nan, np.nan]

    mean_sigma_nn = float(np.nanmean(beta_means))
    se_sigma_nn = float(np.nanstd(beta_means, ddof=1)) / np.sqrt(max(len(beta_means), 1))
    lower_nn = mean_sigma_nn - 1.96 * se_sigma_nn
    upper_nn = mean_sigma_nn + 1.96 * se_sigma_nn

    color_histogram = "#6b756f"
    color_shaded_nn = "#5f78b9"
    color_mean_nn = "#5f78b9"

    LARGE_FONT = 22

    plt.figure(figsize=(10, 8))
    sns.histplot(std_vals, bins=30, kde=True, color=color_histogram, alpha=0.2, edgecolor="black", stat="density")
    if not np.isnan(mean_sigma_nn):
        plt.axvspan(lower_nn, upper_nn, color=color_shaded_nn, alpha=0.1)
        plt.axvline(mean_sigma_nn, color=color_mean_nn, linestyle="solid", linewidth=4,
                    label=f"Non-linear model uncertainty:\n{mean_sigma_nn:.1f} [{lower_nn:.1f}, {upper_nn:.1f}]")

    plt.xlabel("Standard deviation from kriging [mol %]", fontsize=LARGE_FONT)
    plt.ylabel("Density", fontsize=LARGE_FONT)
    plt.xticks(fontsize=LARGE_FONT)
    plt.yticks(fontsize=LARGE_FONT)

    plt.legend(fontsize=LARGE_FONT, loc="upper right")
    plt.grid(True, linestyle="dotted", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{data_source}_{comp}_std_comparison.png")
    plt.show()


# =========================================
# Batch load merged tables -> all_data dict
# =========================================
radius_factors = [0.15, 0.175]
data_sources = ['ghgrp', 'usgs']
threshold = 1000
keep_nanapis = True
filter_prod = True
sample_frac = 0.1  # take 10% (comment said 20%; adjust if you want)
random_state = 42

all_data: dict[str, list[pd.DataFrame]] = {}

for data_source in data_sources:
    for radius_factor in tqdm(radius_factors, desc=f'Processing {data_source}'):
        list_suffix = f"{data_source}_r{str(radius_factor).replace('.', '')}"

        merged_kriging_list_C1 = []
        merged_simple_kriging_list_C1 = []
        merged_nn_list_C1 = []

        # C2 lists only for USGS
        merged_kriging_list_C2 = []
        merged_simple_kriging_list_C2 = []
        merged_nn_list_C2 = []

        for a, b in zip([-1, 1, -1, 1], [-1, 1, 1, -1]):
            if data_source == 'usgs':
                # C2 (USGS only)
                merged_kriging_C2 = pd.read_csv(
                    f'{out_path}/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_kriging_C2_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )
                merged_simple_kriging_C2 = pd.read_csv(
                    f'{out_path}/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_simple_kriging_C2_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )
                merged_nn_C2 = pd.read_csv(
                    f'{out_path}/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_nn_C2_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )
                merged_kriging_list_C2.append(merged_kriging_C2)
                merged_simple_kriging_list_C2.append(merged_simple_kriging_C2)
                merged_nn_list_C2.append(merged_nn_C2)

                # C1 as well
                merged_kriging_C1 = pd.read_csv(
                    f'{out_path}/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_kriging_C1_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )
                merged_simple_kriging_C1 = pd.read_csv(
                    f'{out_path}/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_simple_kriging_C1_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )
                merged_nn_C1 = pd.read_csv(
                    f'{out_path}/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_nn_C1_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )

                merged_kriging_list_C1.append(merged_kriging_C1)
                merged_simple_kriging_list_C1.append(merged_simple_kriging_C1)
                merged_nn_list_C1.append(merged_nn_C1)

            else:
                # GHGRP only C1
                merged_kriging_C1 = pd.read_csv(
                    f'{out_path}/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_kriging_C1_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )
                merged_simple_kriging_C1 = pd.read_csv(
                    f'{out_path}/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_simple_kriging_C1_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )
                merged_nn_C1 = pd.read_csv(
                    f'{out_path}/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_nn_C1_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )

                # Optional sub-sampling (kept from your code)
                if not merged_kriging_C1.empty:
                    subset_idx = merged_kriging_C1.sample(frac=sample_frac, random_state=random_state).index
                    merged_kriging_C1 = merged_kriging_C1.loc[subset_idx].reset_index(drop=True)
                    merged_simple_kriging_C1 = merged_simple_kriging_C1.loc[subset_idx].reset_index(drop=True)
                    merged_nn_C1 = merged_nn_C1.loc[subset_idx].reset_index(drop=True)

                merged_kriging_list_C1.append(merged_kriging_C1)
                merged_simple_kriging_list_C1.append(merged_simple_kriging_C1)
                merged_nn_list_C1.append(merged_nn_C1)

        # Store lists
        all_data[f'merged_kriging_list_C1_{list_suffix}'] = merged_kriging_list_C1
        all_data[f'merged_simple_kriging_list_C1_{list_suffix}'] = merged_simple_kriging_list_C1
        all_data[f'merged_nn_list_C1_{list_suffix}'] = merged_nn_list_C1

        if data_source == 'usgs':
            all_data[f'merged_kriging_list_C2_{list_suffix}'] = merged_kriging_list_C2
            all_data[f'merged_simple_kriging_list_C2_{list_suffix}'] = merged_simple_kriging_list_C2
            all_data[f'merged_nn_list_C2_{list_suffix}'] = merged_nn_list_C2


# =========================
# Example calls & figures
# =========================

# Distance/variance figures for GHGRP (C1)
data_source = 'ghgrp'
list_suffix = f'{data_source}_r015'
plot_comparison_lists_v4(
    data_source=data_source,
    nbins=3,
    threshold=0.5,
    comp='C1',
    merged_simple_kriging_list=all_data[f'merged_simple_kriging_list_C1_{list_suffix}'],
    merged_kriging_list=all_data[f'merged_kriging_list_C1_{list_suffix}'],
    merged_nn_list=all_data[f'merged_nn_list_C1_{list_suffix}']
)

# Distance/variance figures for USGS (C1)
data_source = 'usgs'
list_suffix = f'{data_source}_r0175'
plot_comparison_lists_v4(
    data_source=data_source,
    nbins=3,
    threshold=0.8,
    comp='C1',
    merged_simple_kriging_list=all_data[f'merged_simple_kriging_list_C1_{list_suffix}'],
    merged_kriging_list=all_data[f'merged_kriging_list_C1_{list_suffix}'],
    merged_nn_list=all_data[f'merged_nn_list_C1_{list_suffix}'],
    propagate=False
)

# Std comparison figures
comp = 'C1'
data_source = 'usgs'
list_suffix = f'{data_source}_r0175'
run_comparison_std(data_source, comp, list_suffix)

comp = 'C2'
data_source = 'usgs'
list_suffix = f'{data_source}_r0175'
run_comparison_std(data_source, comp, list_suffix)

comp = 'C1'
data_source = 'ghgrp'
list_suffix = f'{data_source}_r015'
run_comparison_std(data_source, comp, list_suffix)
