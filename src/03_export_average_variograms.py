#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Average variograms across basins (USGS or GHGRP) on a space–time grid.

Pipeline
--------
1) Load per-basin production (for masking).
2) Build a 3D grid (X, Y, T) per basin and aggregate composition & production to that grid.
3) For each component (HE, CO2, C1, ...):
   a) Quantile-normalize values (to N(0,1)).
   b) Stack (coords, values) across selected basins (masked to cells with Gas > 0).
   c) Fit an (isotropic) variogram with scikit-gstat.
   d) Save parameters to JSON and plot the fitted exponential model.

Notes
-----
- Spatial coordinates X,Y are rescaled by alpha before gridding (space–time anisotropy).
- skgstat `Variogram` supports nD coordinates; we pass (X, Y, T).
"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import skgstat as skg
from skgstat import models

# ------------------------------ #
#           CONFIG PATH          #
# ------------------------------ #

root_path = '/scratch/users/pburdeau/data/gas_composition'


# ------------------------------ #
#         GRID UTILITIES         #
# ------------------------------ #

def create_grid(df: pd.DataFrame, res_space: float, res_time: int):
    """
    Build a regular 3D grid covering the extent of df for (X, Y, T).

    Returns
    -------
    grid : (N, 3) ndarray, stacked ij meshgrid points (X,Y,T)
    grid_params : tuple
        (xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time)
    """
    xmin, xmax = df['X'].min(), df['X'].max()
    ymin, ymax = df['Y'].min(), df['Y'].max()
    tmin, tmax = df['T'].min(), df['T'].max()
    grid_params = (xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time)

    x_range = np.arange(xmin, xmax + res_space, res_space)
    y_range = np.arange(ymin, ymax + res_space, res_space)
    t_range = np.arange(tmin, tmax + res_time, res_time)

    grid_coord = np.meshgrid(x_range, y_range, t_range, indexing='ij')
    grid = np.column_stack([g.ravel() for g in grid_coord])
    return grid, grid_params


def aggregate(df: pd.DataFrame, list_zz, grid_params, mean_agg: bool = True) -> pd.DataFrame:
    """
    Aggregate df values onto the regular 3D grid by (X, Y, T) bins.

    Parameters
    ----------
    df : DataFrame with columns ['X','Y','T', <variables ...>]
    list_zz : list[str] variables to aggregate
    grid_params : tuple returned by create_grid
    mean_agg : if True, compute mean within each voxel; else sum

    Returns
    -------
    df_grid : DataFrame with columns ['X','Y','T', ...list_zz]
    """
    xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time = grid_params

    x_range = np.arange(xmin, xmax + res_space, res_space)
    y_range = np.arange(ymin, ymax + res_space, res_space)
    t_range = np.arange(tmin, tmax + res_time, res_time)

    # keep only data falling within the grid bounds
    df = df[(df.X >= xmin) & (df.X <= xmax) &
            (df.Y >= ymin) & (df.Y <= ymax) &
            (df['T'] >= tmin) & (df['T'] <= tmax)].reset_index(drop=True)

    # base output: all grid cells
    grid_coord = np.meshgrid(x_range, y_range, t_range, indexing='ij')
    flat_grid_coords = np.column_stack([g.ravel() for g in grid_coord])
    df_grid = pd.DataFrame(flat_grid_coords, columns=['X', 'Y', 'T'])

    # aggregate each variable onto the grid
    for zz in list_zz:
        list_cols = ['X', 'Y', 'T', zz]
        np_data = df[list_cols].to_numpy().astype(float)

        origin = np.array([xmin, ymin, tmin], dtype=float)
        resolution = np.array([res_space, res_space, res_time], dtype=float)

        grid_indices = np.rint(((np_data[:, :3] - origin) / resolution)).astype(int)
        grid_shape = (len(x_range), len(y_range), len(t_range))

        grid_sum = np.zeros(grid_shape, dtype=float)
        grid_count = np.zeros(grid_shape, dtype=float)

        # accumulate values per voxel
        for i in range(np_data.shape[0]):
            x_idx, y_idx, t_idx = grid_indices[i, :3]
            if (0 <= x_idx < grid_shape[0]) and (0 <= y_idx < grid_shape[1]) and (0 <= t_idx < grid_shape[2]):
                grid_sum[x_idx, y_idx, t_idx] += np_data[i, 3]
                grid_count[x_idx, y_idx, t_idx] += 1

        if mean_agg:
            with np.errstate(invalid='ignore', divide='ignore'):
                grid_matrix = np.divide(grid_sum, grid_count)
        else:
            grid_matrix = grid_sum

        df_grid[zz] = grid_matrix.ravel()

    return df_grid


# ------------------------------ #
#         VARIOGRAM UTILS        #
# ------------------------------ #

def create_variogram(coords, values, n_lags: int, maxlag: float):
    """
    Fit a (default isotropic) variogram in scikit-gstat.

    Returns
    -------
    azimuth, nugget, major_range, minor_range, sill, vtype
    """
    V1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags,
                       maxlag=maxlag, normalize=False)
    # scikit-gstat parameter order: [range, sill, nugget]
    azimuth = 0
    nugget = V1.parameters[2]
    major_range = V1.parameters[0]
    minor_range = V1.parameters[0]
    sill = V1.parameters[1]
    vtype = 'Exponential'
    return azimuth, nugget, major_range, minor_range, sill, vtype


def plot_variogram(V1: skg.Variogram, title: str, maxlag: float, output_plot: str):
    """
    Plot experimental vs fitted exponential variogram.
    (Matches your styling and alpha scaling used elsewhere.)
    """
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)

    xdata = V1.bins
    ydata = V1.experimental
    sill = V1.parameters[1]
    range_val = V1.parameters[0]

    V1.model = 'exponential'
    xi = np.linspace(0, xdata[-1], 100)
    y_exp = [models.exponential(h, V1.parameters[0], V1.parameters[1], V1.parameters[2]) for h in xi]

    fig, ax = plt.subplots(figsize=(8, 6))
    # scale x to km using the study’s alpha (6.34) as in your original plotting
    ax.plot(xdata * 6.34 / 1000, ydata, 'o', label="Experimental variogram", color='#adc2da')
    ax.plot(xi * 6.34 / 1000, y_exp, '-', label='Exponential variogram', color='#3a4659')

    # Annotate sill & range (range also scaled)
    text_x = max(xi) * 6.34 / 1000 * 0.7
    text_y = max(y_exp) * 0.8
    ax.text(text_x, text_y, f"Sill: {sill:.1f}\nRange: {range_val * 6.34 / 1000:.1f} km",
            fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Lag [km]', fontsize=14)
    ax.set_ylabel('Semivariance', fontsize=14)
    ax.legend(loc='lower right', fontsize=14)

    plt.savefig(output_plot)
    plt.show()


# ------------------------------ #
#     AVERAGE VARIOGRAM CORE     #
# ------------------------------ #

def average_variogram(basins_for_average_variogram,
                      data_source: str,
                      df_prod: pd.DataFrame,
                      res_space: float,
                      res_time: int,
                      alpha: float,
                      maxlag: float,
                      n_lags: int,
                      keep_nanapis: bool,
                      threshold: float):
    """
    Build average variograms across basins for all available components.

    Parameters
    ----------
    data_source : 'usgs' or 'ghgrp'
    keep_nanapis : bool
        Whether to use the USGS file with NaN APIs kept (your two preprocessed versions).
    threshold : float
        Non-hydrocarbon sum threshold (applied to USGS only).
    """
    # Load data source
    if data_source == 'usgs':
        if keep_nanapis:
            data = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_with_nanapis.csv'))
        else:
            data = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_without_nanapis.csv'))

        # Filter by non-hydrocarbon total
        def filter_non_hydrocarbons(df, thr):
            non_hc = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']
            df['non_hydrocarbon_sum'] = df[non_hc].fillna(0).sum(axis=1)
            return df[df['non_hydrocarbon_sum'] < thr].drop(columns=['non_hydrocarbon_sum'])

        data = filter_non_hydrocarbons(data, threshold)

    else:
        data = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))

    # Output directory per setting
    output_dir = os.path.join(
        root_path, 'average_variograms',
        f'{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_1year'
    )
    os.makedirs(output_dir, exist_ok=True)

    # Subset basins of interest, rescale space by alpha for anisotropy
    df_data = data.reset_index(drop=True)
    df_data = df_data[df_data.BASIN_NAME.isin(basins_for_average_variogram)].reset_index(drop=True)
    df_data['X'] = df_data['X'] / alpha
    df_data['Y'] = df_data['Y'] / alpha

    # Components present in df_data
    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2',
                  'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
    components = [c for c in components if c in df_data.columns]
    if not components:
        raise ValueError("No valid components found in df_data columns.")

    # Iterate over components
    for comp in components:
        print(f"Processing component: {comp}")

        combined_coords = []
        combined_values = []

        for basin in basins_for_average_variogram:
            basin_data = df_data[df_data.BASIN_NAME == basin].reset_index(drop=True)

            # Grid for this basin
            grid, grid_params = create_grid(df_prod[df_prod.BASIN_NAME == basin], res_space, res_time)

            # Production on grid (sum)
            prod_on_grid = aggregate(df_prod[df_prod.BASIN_NAME == basin], ['Gas'], grid_params, mean_agg=False)

            # Component on grid (mean)
            data_on_grid = aggregate(basin_data, [comp], grid_params, mean_agg=True)

            # Mask to where Gas > 0 (active production)
            data_on_grid = data_on_grid[prod_on_grid.Gas > 0].reset_index(drop=True)

            # Keep positive, non-NaN component values
            data_single = data_on_grid[~pd.isna(data_on_grid[comp])].reset_index(drop=True)
            data_single = data_single[data_single[comp] > 0].reset_index(drop=True)

            if not data_single.empty:
                # Normal-score transform to stabilize variance
                transformer = QuantileTransformer(n_quantiles=500, output_distribution='normal')
                data_single['norm_' + comp] = transformer.fit_transform(
                    data_single[comp].values.reshape(-1, 1)
                ).flatten()

                coords = data_single[['X', 'Y', 'T']].values
                values = data_single['norm_' + comp].values

                combined_coords.append(coords)
                combined_values.append(values)

        # Combine all basins for this component
        if combined_coords:
            all_coords = np.vstack(combined_coords)
            all_values = np.hstack(combined_values)

            # Fit variogram and save parameters
            azimuth, nugget, major_range, minor_range, sill, vtype = create_variogram(
                all_coords, all_values, n_lags, maxlag
            )
            variogram_data = {
                "azimuth": azimuth,
                "nugget": nugget,
                "major_range": major_range,
                "minor_range": minor_range,
                "sill": sill,
                "type": vtype
            }

            variogram_file = os.path.join(output_dir, f'{comp}_variogram.json')
            with open(variogram_file, 'w') as f:
                json.dump(variogram_data, f, indent=4)
            print(f"Saved variogram for {comp} to {variogram_file}")

            # Plot and save experimental + model variogram
            V_for_plot = skg.Variogram(all_coords, all_values, bin_func='even', n_lags=n_lags, maxlag=maxlag)
            output_plot = os.path.join(output_dir, f'{comp}_variogram.pdf')
            plot_variogram(V_for_plot, f"Combined Variogram for {comp}", maxlag, output_plot)

    print(f"All variograms saved in {output_dir}")


# ------------------------------ #
#           ENTRYPOINT           #
# ------------------------------ #

def my_function(data_source: str, keep_nanapis: bool, threshold: float):
    """
    Driver wrapper: load production, set grid+variogram params, and call average_variogram.
    """
    basins_for_average_variogram = [
        'Appalachian Basin',
        'Appalachian Basin (Eastern Overthrust Area)',
        'Permian Basin',
        'Arkla Basin',
        'Anadarko Basin',
        'Denver Basin',
        'Green River Basin',
        'Arkoma Basin',
        'Gulf Coast Basin (LA, TX)',
        'East Texas Basin'
    ]

    base_path = os.path.join(root_path, 'wells_info_prod_per_basin')

    # Concatenate per-basin production (already aggregated in your earlier pipeline)
    combined_data = pd.DataFrame()
    for basin in basins_for_average_variogram:
        file_path = os.path.join(base_path, f'{basin}_final.csv')
        print(f"Reading data for basin: {basin}")
        basin_data = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, basin_data], ignore_index=True)

    # Space–time scaling and gridding params
    alpha = 6.34
    maxlag = 100000 / alpha   # meters (after spatial rescale)
    n_lags = 10
    k = 100                  # (not used directly here; kept for context)
    rad = maxlag             # (not used directly here; kept for context)
    res_space = 2000 / alpha # meters
    res_time = 365           # days (1 year)

    # Production preprocessing & rescale space
    df_prod = combined_data.reset_index(drop=True)
    df_prod = df_prod.rename(columns={'Monthly Oil': 'Oil', 'Monthly Gas': 'Gas'})
    df_prod = df_prod[df_prod.Gas > 0].reset_index(drop=True)
    df_prod['Oil'].fillna(0, inplace=True)
    df_prod['X'] = df_prod['X'] / alpha
    df_prod['Y'] = df_prod['Y'] / alpha
    df_prod = df_prod[~pd.isna(df_prod['Gas'])].reset_index(drop=True)

    # Run the average-variogram workflow
    average_variogram(basins_for_average_variogram, data_source, df_prod,
                      res_space, res_time, alpha, maxlag, n_lags, keep_nanapis, threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Average variogram builder for USGS/GHGRP.")
    parser.add_argument('--data_source', type=int, required=True,
                        help="0=usgs, 1=ghgrp")
    parser.add_argument('--threshold', type=int, required=True,
                        help="Index into thresholds list (see code).")
    parser.add_argument('--keep_nanapis', type=int, required=True,
                        help="0=False, 1=True (USGS only)")

    args = parser.parse_args()

    data_sources = ['usgs', 'ghgrp']
    data_source = data_sources[args.data_source]

    # thresholds ordering matches your original (kept as-is)
    thresholds = [1000, 25, 50, 10]
    threshold = thresholds[args.threshold]

    keep_nanapiss = [True, False]
    keep_nanapis = keep_nanapiss[args.keep_nanapis]

    my_function(data_source, keep_nanapis, threshold)
