#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semivariance/anisotropy workflow:
- Build pairwise spatiotemporal distances and value diffs (per basin, component).
- Bin by space/time for several alpha (space-time scaling) values.
- Assemble an empirical variogram surface and estimate space–time anisotropy alpha
  by fitting a linear model on (near-)purely spatial lags and minimizing RMSE.
"""

# =============== Imports & setup ===============

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import QuantileTransformer
from scipy.spatial.distance import pdist, squareform
from statsmodels.formula.api import ols
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.ticker as mticker
import seaborn as sns

# Ensure figure output dir exists (your plots save into it)
os.makedirs('figures_out', exist_ok=True)


# =============== Core helpers ===============

def find_rmse(empVgm, stAni):
    """
    Compute RMSE between:
      - linear fit gamma ~ timelag using near-pure spatial lags (spacelag <= 5000),
      - empirical variogram points with timelag <= 365,
    after rescaling spatial lag by stAni: timelag_hat = spacelag / stAni.

    Parameters
    ----------
    empVgm : pd.DataFrame with columns ['spacelag', 'timelag', 'gamma']
    stAni : float (space-time anisotropy scaling)

    Returns
    -------
    float or np.nan
    """
    lm_data = empVgm[empVgm['spacelag'] <= 5000]
    if len(lm_data) > 10:
        lm_formula = 'gamma ~ timelag'
        lm_model = ols(lm_formula, data=lm_data).fit()

        filtered_data = empVgm[empVgm['timelag'] <= 365]
        dist_adjusted = filtered_data['spacelag'] / stAni
        predicted_gamma = lm_model.predict(exog=dict(timelag=dist_adjusted))
        rmse = np.sqrt(np.mean((predicted_gamma - filtered_data['gamma'])**2))
        return rmse
    else:
        return np.nan


def compute_diffs_time_space_basin(basin, comp, alpha):
    """
    For a given basin and component:
    - Transform component values to normal scores (quantile transform).
    - Build pairwise spatial distances (meters in your CRS), temporal distances (days),
      combined spatiotemporal distance (sqrt(ds^2 + (alpha^2 * dt^2))),
      and squared diffs of transformed values.

    Returns flattened upper triangles for efficiency.
    """
    usgs_basin_gdf = usgs_gdf_main_basins[usgs_gdf_main_basins.BASIN_NAME == basin]
    usgs_basin_gdf = usgs_basin_gdf[~pd.isna(usgs_basin_gdf[comp])]

    # Coordinates and values
    points = list(zip(usgs_basin_gdf['X'], usgs_basin_gdf['Y']))
    values = usgs_basin_gdf[comp].astype(float).to_numpy().reshape(-1, 1)

    if len(values) > 10:
        # Normal scores transform to stabilize variance
        nst_trans = QuantileTransformer(n_quantiles=500, output_distribution='normal').fit(values)
        values_trans = nst_trans.transform(values)
        values = values_trans.reshape(-1)
        del values_trans

        epochs = usgs_basin_gdf['T'].to_numpy()
        years = usgs_basin_gdf['Year'].to_numpy()  # noqa: F841 (kept for context)
        num_points = len(points)

        # Pairwise spatial distances (meters)
        distances_space = pdist(points)
        distances_space = squareform(distances_space)

        # Pairwise temporal distances (days)
        alpha_squared = alpha ** 2
        distances_time = np.abs(epochs[:, None] - epochs[None, :])

        # Combined spatiotemporal distance (not used later, but computed for context)
        distances = np.sqrt(distances_space**2 + (alpha_squared * distances_time**2))

        # Squared differences of transformed values
        diffs = np.abs(values[:, None] - values[None, :]) ** 2

        # Flatten upper triangle (i<j) to avoid duplicates and self-pairs
        mask_upper_triangle = np.triu(np.ones(num_points, dtype=bool), k=1)
        diffs = diffs[mask_upper_triangle]
        distances = distances[mask_upper_triangle]
        distances_time = distances_time[mask_upper_triangle]
        distances_space = distances_space[mask_upper_triangle]

        return diffs, distances, distances_time, distances_space
    else:
        # Keep types consistent for concatenation upstream
        return [0], [0], [0], [0]


def compute_variances_and_bins(diffs, distances_time, distances_space, alphas, n_bins=6):
    """
    Bin pairs by time/space (rescaled by alpha) and compute mean squared diffs per bin.
    Returns per-alpha 3D array: (time_bins-1, space_bins-1, 3), with the first slice used.
    """
    variances_by_alpha = {}

    for alpha in alphas:
        if alpha < 1:
            # Time bins are scaled down for small alpha
            time_bins = np.linspace(np.min(distances_time) * alpha / 4., np.max(distances_time) * alpha / 4., n_bins)
            space_bins = np.linspace(np.min(distances_space) / 4., np.max(distances_space) / 4., n_bins)
            scaled_space_bins = time_bins  # <-- ensure defined in both branches
        else:
            time_bins = np.linspace(np.min(distances_time) / 4., np.max(distances_time) / 4., n_bins)
            scaled_space_bins = time_bins
            space_bins = np.linspace(np.min(distances_space) / (4. * alpha), np.max(distances_space) / (4. * alpha), n_bins)

        # Rescaled distances (kept for clarity; indices below reflect the intended scaling)
        adjusted_time_distances = distances_time * alpha     # noqa: F841
        adjusted_space_distances = distances_space * alpha   # noqa: F841

        # Digitize into bins; space uses the "scaled" counterpart per your method
        time_indices = np.digitize(distances_time, time_bins) - 1
        space_indices = np.digitize(np.array(distances_space) / alpha, scaled_space_bins) - 1

        variances = np.zeros((len(time_bins) - 1, len(space_bins) - 1, 3))

        for i in range(len(time_bins) - 1):
            for j in range(len(space_bins) - 1):
                mask_original = (time_indices == i) & (space_indices == j)
                if np.any(mask_original):
                    variances[i, j, 0] = np.mean(diffs[mask_original])

        variances_by_alpha[alpha] = variances

    return time_bins, scaled_space_bins, variances_by_alpha


def compute_variances_and_bins_for_est(diffs, distances_time, distances_space, alphas, n_bins=6):
    """
    Variant used for the estimator workflow; bins without the scaled-space indirection.
    Returns (time_bins, space_bins, variances_by_alpha).
    """
    variances_by_alpha = {}

    for alpha in alphas:
        if alpha < 1:
            time_bins = np.linspace(np.min(distances_time) * alpha / 4., np.max(distances_time) * alpha / 4., n_bins)
            space_bins = np.linspace(np.min(distances_space) / 4., np.max(distances_space) / 4., n_bins)
        else:
            time_bins = np.linspace(np.min(distances_time) / 4., np.max(distances_time) / 4., n_bins)
            space_bins = np.linspace(np.min(distances_space) / (4. * alpha), np.max(distances_space) / (4. * alpha), n_bins)

        adjusted_time_distances = distances_time * alpha     # noqa: F841 (kept for clarity)
        adjusted_space_distances = distances_space * alpha   # noqa: F841

        time_indices = np.digitize(distances_time, time_bins) - 1
        space_indices = np.digitize(distances_space, space_bins) - 1

        variances = np.zeros((len(time_bins) - 1, len(space_bins) - 1, 3))
        for i in range(len(time_bins) - 1):
            for j in range(len(space_bins) - 1):
                mask_original = (time_indices == i) & (space_indices == j)
                if np.any(mask_original):
                    variances[i, j, 0] = np.mean(diffs[mask_original])

        variances_by_alpha[alpha] = variances

    return time_bins, space_bins, variances_by_alpha


def create_dataframe_from_variances(variances_by_alpha, space_bins, time_bins, alphas):
    """
    Create a tidy DataFrame of empirical semivariances per (space/time bin center).

    This function supports two input shapes:
    (A) variances_by_alpha: 2D numpy array [time_bin, space_bin]  (your current call passes this)
    (B) variances_by_alpha: dict {alpha: 3D array [time_bin, space_bin, k]}, we will use k=0

    Returns
    -------
    pd.DataFrame with columns ['spacelag','timelag','gamma']
    """
    # Bin centers for readability
    space_bin_centers = (np.asarray(space_bins[:-1]) + np.asarray(space_bins[1:])) / 2
    time_bin_centers  = (np.asarray(time_bins[:-1]) + np.asarray(time_bins[1:])) / 2

    records = []

    # Case (A): 2D array
    if isinstance(variances_by_alpha, np.ndarray) and variances_by_alpha.ndim == 2:
        Z = variances_by_alpha
        for i, t_center in enumerate(time_bin_centers):
            for j, s_center in enumerate(space_bin_centers):
                records.append({
                    'spacelag': s_center,
                    'timelag': t_center,
                    'gamma': float(Z[i, j])
                })
    else:
        # Case (B): dict-like (not used in your immediate call, but supported for completeness)
        for alpha in alphas:
            Z = variances_by_alpha[alpha][..., 0]
            for i, t_center in enumerate(time_bin_centers):
                for j, s_center in enumerate(space_bin_centers):
                    records.append({
                        'spacelag': s_center,
                        'timelag': t_center,
                        'gamma': float(Z[i, j])
                    })

    return pd.DataFrame(records)


def estiStAni_lin_space(empVgm, interval, figures_path, colormap):
    """
    Estimate space–time anisotropy (alpha) by:
      1) fitting gamma ~ timelag on near-pure spatial lags (spacelag <= 5000),
      2) finding stAni that minimizes RMSE when mapping spacelag -> timelag via spacelag/alpha.

    Also produces diagnostic plots at intermediate iterations and for selected alphas.
    """
    os.makedirs(figures_path, exist_ok=True)

    lm_data = empVgm[empVgm['spacelag'] <= 5000]
    if len(lm_data) > 10:
        lm_formula = 'gamma ~ timelag'
        lm_model = ols(lm_formula, data=lm_data).fit()

        def optFun(stAni, empVgm, plot=False, label=None, colormap=None):
            filtered_data = empVgm[empVgm['timelag'] <= 365]
            dist_adjusted = filtered_data['spacelag'] / stAni
            predicted_gamma = lm_model.predict(exog=dict(timelag=dist_adjusted))
            rmse = np.sqrt(np.mean((predicted_gamma - filtered_data['gamma'])**2))

            if plot:
                stAni_scalar = stAni[0] if isinstance(stAni, (np.ndarray, list)) else stAni
                fig, ax1 = plt.subplots(figsize=(11, 8))
                plt.rcParams.update({'font.size': 26})

                # Fixed color selection from colormap
                cmap = colormap or sns.cubehelix_palette(start=2.1, hue=1.7, light=0.9, dark=0.2,
                                                         rot=0.4, as_cmap=True, reverse=True)
                colors = [cmap(0.2), cmap(0.5)]

                # Empirical variogram points
                ax1.scatter(
                    filtered_data['spacelag'] / 1000,  # km
                    filtered_data['gamma'],
                    label=r"Empirical spatial variogram ($\gamma_s$)",
                    color=colors[0],
                    s=50
                )
                ax1.set_xlabel(r"Spatial distance ($D_s$) [km]")
                ax1.set_ylabel("Semivariance [mol %$^2$]")

                # Predicted line using spacelag/alpha
                ax1.scatter(
                    filtered_data['spacelag'] / 1000,
                    predicted_gamma,
                    label=r"$\beta_0 + \beta_1 \cdot \frac{D_s}{\alpha}$",
                    color=colors[1],
                    alpha=0.6,
                    s=50
                )

                ax1.set_title(f"$\\alpha = {stAni_scalar:.1f}$, RMSE: {rmse:.1f} mol %$^2$\n")
                legend = ax1.legend(loc="upper left", fontsize=26)
                legend.get_frame().set_alpha(0.5)

                filename = f"plot_stAni_{label}.eps" if label else f"plot_stAni_{stAni_scalar:.2f}.eps"
                plt.tight_layout()
                plt.savefig(os.path.join(figures_path, filename), format='eps', dpi=300)
                plt.show()
                plt.close(fig)

            return rmse

        # Wrapper that occasionally plots during optimization
        iteration_count = [0]
        def optFunWrapper(stAni, *args):
            rmse = optFun(stAni, *args)
            if iteration_count[0] % 10 == 0:
                optFun(stAni, *args, plot=True)
            iteration_count[0] += 1
            return rmse

        # Nelder–Mead over alpha in the supplied interval
        result = minimize(
            fun=optFunWrapper,
            x0=[np.mean(interval)],
            args=(empVgm,),
            method='Nelder-Mead',
            options={'xatol': 1e-1, 'fatol': 1e-10}
        )

        # Diagnostic plots at selected alphas
        optFun(result.x[0], empVgm, plot=True, label="optimal")
        optFun(1,           empVgm, plot=True, label="1")
        optFun(40,          empVgm, plot=True, label="40")

        return result.x[0]
    else:
        return np.nan


def plot_2d(alphas, diffs, distances_time, distances_space, nbins=6,
            cubehelix_custom=sns.cubehelix_palette(start=0.2, hue=1.4, light=0.8, dark=0.3, as_cmap=True, reverse=True)):
    """
    Heatmap + contour of variance surface for the first alpha in `alphas`.
    """
    vmin, vmax = 0, 3  # fixed color scale
    fontsize = 26
    alpha = alphas[0]

    time_bins, space_bins, variances_by_alpha = compute_variances_and_bins(
        diffs, distances_time, distances_space, alphas, nbins
    )
    space_bins_sorted = np.sort(space_bins)
    time_bins_sorted = np.sort(time_bins)

    space_centers = (space_bins_sorted[:-1] + space_bins_sorted[1:]) / 2
    time_centers = (time_bins_sorted[:-1] + time_bins_sorted[1:]) / 2

    X, Y = np.meshgrid(space_centers, time_centers)
    Z = variances_by_alpha[alpha][..., 0]

    # Common limits/ticks across both axes for square aspect
    common_limit = max(space_centers[-1], time_centers[-1])
    common_ticks = np.linspace(0, common_limit, 6)

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 7))
    heatmap = ax.imshow(
        Z, cmap=cubehelix_custom, aspect='auto', origin='lower',
        extent=[0, common_limit, 0, common_limit]
    )
    ax.set_ylabel('Time difference [days]', fontsize=fontsize)
    ax.set_xlabel(f'Scaled space difference [m / {alpha}]', fontsize=fontsize)
    ax.set_title(rf'$\alpha$ = {alpha}', fontsize=fontsize)
    fig.colorbar(heatmap, ax=ax, label='Variance')

    ax.set_xticks(common_ticks)
    ax.set_yticks(common_ticks)

    def format_ticks(value, _):
        if value == 0:
            return "0"
        power = int(np.floor(np.log10(abs(value))))
        factor = int(value / (10 ** power))
        return f"{factor}e{power}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(1000, 7000)

    plt.savefig(f'figures_out/heatmap_alpha_{alpha:.1f}.eps', format='eps', dpi=300)
    plt.show()

    # --- Contour map ---
    fig, ax = plt.subplots(figsize=(10, 7))
    contour = ax.contourf(X, Y, Z, levels=np.linspace(vmin, vmax, 25), cmap=cubehelix_custom, vmin=vmin, vmax=vmax)
    lines = ax.contour(X, Y, Z, levels=np.linspace(vmin, vmax, 5), colors='black', linewidths=0.5)
    ax.clabel(lines, inline=True, fontsize=fontsize - 6, fmt="%.1f")

    ax.set_ylabel('Temporal lag [days]', fontsize=fontsize)
    ax.set_xlabel(rf'Rescaled spatial lag [days]', fontsize=fontsize)
    ax.set_title(rf'$\alpha$ = {alpha:.1f}', fontsize=fontsize)

    ax.set_xticks(common_ticks)
    ax.set_yticks(common_ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_ticks))
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xlim(1000, 7000)
    ax.set_ylim(1000, 7000)
    ax.set_aspect('equal', 'box')

    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label(r'Variance [mol %$^2$]', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)
    num_ticks = 5
    tick_values = np.linspace(vmin, vmax, num_ticks)
    cbar.set_ticks(tick_values)
    cbar.ax.set_yticklabels([f'{tick:.1f}' for tick in tick_values])

    plt.savefig(f'figures_out/contourmap_alpha_{alpha:.1f}.eps', format='eps', dpi=300)
    plt.show()


# =============== Data loading & pipeline ===============

root_path = '/scratch/users/pburdeau/data/gas_composition'
usgs_data = pd.read_csv(os.path.join(root_path, 'usgs/usgs_processed_with_nanapis.csv'))

basins_to_keep = [
    'Appalachian Basin', 'Appalachian Basin (Eastern Overthrust Area)', 'Permian Basin', 'Arkla Basin',
    'Anadarko Basin', 'San Joaquin Basin', 'Denver Basin', 'Uinta Basin', 'Green River Basin',
    'Arkoma Basin', 'Gulf Coast Basin (LA, TX)', 'Williston Basin', 'East Texas Basin'
]

global usgs_gdf_main_basins
usgs_gdf_main_basins = usgs_data[usgs_data.BASIN_NAME.isin(basins_to_keep)].reset_index(drop=True)

comp = 'C1'
alpha = 1

all_diffs = []
all_distances_time = []
all_distances_space = []
all_distances = []

# Build pooled pairwise stats across selected basins
for basin in basins_to_keep:
    diffs, distances, distances_time, distances_space = compute_diffs_time_space_basin(basin, comp, alpha)
    all_diffs = np.concatenate([all_diffs, diffs])
    all_distances_time = np.concatenate([all_distances_time, distances_time])
    all_distances_space = np.concatenate([all_distances_space, distances_space])
    all_distances = np.concatenate([all_distances, distances])

# Bin pairs (estimation variant) and assemble tidy DF
alphas = [1]
time_bins, space_bins, variances_by_alpha = compute_variances_and_bins_for_est(
    all_diffs, all_distances_time, all_distances_space, alphas, 50
)

df_variances = create_dataframe_from_variances(
    variances_by_alpha[alphas[0]][..., 0],  # 2D array [time, space]
    space_bins, time_bins, alphas
)

# Quick diagnostic scatters
plt.scatter(df_variances[df_variances.spacelag <= 5000].timelag,
            df_variances[df_variances.spacelag <= 5000].gamma)
plt.xlim(0, 10000)
plt.show()

plt.scatter(df_variances[df_variances.timelag <= 200].spacelag,
            df_variances[df_variances.timelag <= 200].gamma)
plt.show()

# Estimate anisotropy parameter alpha on empirical surface
interval = [0.1, 100]
estiStAni_lin_space(
    df_variances,
    interval,
    figures_path='figures_out',
    colormap=sns.cubehelix_palette(start=2.1, hue=1.7, light=0.9, dark=0.2, rot=0.4, as_cmap=True, reverse=True)
)

# 2D variance surfaces for selected alphas
for alpha_val in [1, 6.34, 40]:
    plot_2d([alpha_val], all_diffs, all_distances_time, all_distances_space, nbins=6)
