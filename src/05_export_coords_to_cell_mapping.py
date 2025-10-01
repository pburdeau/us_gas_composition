#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build per-point -> grid cell index maps for composition and production (Gas/Oil).

What this script does
---------------------
1) Load composition data (USGS or GHGRP) with an optional non-HC filter.
2) For each basin that has a production file:
   - Rescale X,Y by alpha (space–time anisotropy) to build a common (X,Y,T) grid.
   - Aggregate composition (mean) and production (sum) to that grid.
   - Record, for each raw row, the index of the grid voxel it falls into and the voxel's
     (X_grid, Y_grid, T_grid) coordinates.
3) After looping basins, merge/sum production duplicates and rescale X,Y back to original units.

Returns
-------
data_indices_all       : DataFrame of per-record → grid-cell indices for all composition components
all_prod_indices_all   : DataFrame of per-record → grid-cell indices for production (Gas & Oil), summed
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

root_path = '/scratch/users/pburdeau/data/gas_composition'

# ------------------------------ #
#            GRIDDING            #
# ------------------------------ #

def create_grid(df: pd.DataFrame, res_space: float, res_time: int):
    """
    Create a regular (X,Y,T) grid covering df extents with inclusive upper bounds.

    Returns
    -------
    grid : (N,3) ndarray of [X,Y,T] voxel centers
    grid_params : tuple (xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time)
    """
    xmin, xmax = df['X'].min(), df['X'].max()
    ymin, ymax = df['Y'].min(), df['Y'].max()
    tmin, tmax = df['T'].min(), df['T'].max()
    grid_params = (xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time)

    x_range = np.arange(xmin, xmax + res_space, res_space)
    y_range = np.arange(ymin, ymax + res_space, res_space)
    t_range = np.arange(tmin, tmax + res_time, res_time)

    xx, yy, tt = np.meshgrid(x_range, y_range, t_range, indexing='ij')
    grid = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])
    return grid, grid_params


def aggregate(df: pd.DataFrame, list_zz, grid_params, mean_agg: bool = True):
    """
    Aggregate df values onto the (X,Y,T) grid and record per-row voxel indices/centers.

    Parameters
    ----------
    df        : DataFrame containing ['X','Y','T', ...list_zz]
    list_zz   : list of columns to aggregate
    grid_params: tuple from create_grid
    mean_agg  : if True, compute mean within voxel; else sum

    Returns
    -------
    df_grid   : DataFrame of grid cells with aggregated values for each zz
    indices_df: DataFrame with original per-row (X,Y,T), each zz value, its voxel index,
                and the voxel center coordinates for that zz.
    """
    xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time = grid_params
    x_range = np.arange(xmin, xmax + res_space, res_space)
    y_range = np.arange(ymin, ymax + res_space, res_space)
    t_range = np.arange(tmin, tmax + res_time, res_time)

    # Keep only rows inside grid extents
    df = df[(df.X >= xmin) & (df.X <= xmax) &
            (df.Y >= ymin) & (df.Y <= ymax) &
            (df['T'] >= tmin) & (df['T'] <= tmax)].reset_index(drop=True)

    # Prepare grid frame
    xx, yy, tt = np.meshgrid(x_range, y_range, t_range, indexing='ij')
    flat_grid_coords = np.column_stack([xx.ravel(), yy.ravel(), tt.ravel()])
    df_grid = pd.DataFrame(flat_grid_coords, columns=['X', 'Y', 'T'])

    # Keep original coordinates + placeholders per-variable
    indices_df = df[['X', 'Y', 'T']].copy()

    for zz in list_zz:
        # Track the original values for reference
        indices_df[zz] = df[zz]

        # Compute voxel indices for each row
        np_data = df[['X', 'Y', 'T', zz]].to_numpy(dtype=float)
        origin = np.array([xmin, ymin, tmin], dtype=float)
        resolution = np.array([res_space, res_space, res_time], dtype=float)

        grid_indices = np.rint(((np_data[:, :3] - origin) / resolution)).astype(int)
        indices_df[f'{zz}_grid_index'] = list(zip(grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]))

        # Aggregate to 3D arrays
        shape = (len(x_range), len(y_range), len(t_range))
        grid_sum = np.zeros(shape, dtype=float)
        grid_count = np.zeros(shape, dtype=float)

        for i in range(np_data.shape[0]):
            x_idx, y_idx, t_idx = grid_indices[i, :3]
            # Bounds guard
            if 0 <= x_idx < shape[0] and 0 <= y_idx < shape[1] and 0 <= t_idx < shape[2]:
                grid_sum[x_idx, y_idx, t_idx] += np_data[i, 3]
                grid_count[x_idx, y_idx, t_idx] += 1

        if mean_agg:
            with np.errstate(invalid='ignore', divide='ignore'):
                grid_matrix = np.divide(grid_sum, grid_count)
        else:
            grid_matrix = grid_sum

        df_grid[zz] = grid_matrix.ravel()

        # Add voxel center coordinates for this variable
        # (index-based mapping back to x/y/t ranges)
        indices_df[f'{zz}_X_grid'] = x_range[np.clip(grid_indices[:, 0], 0, len(x_range) - 1)]
        indices_df[f'{zz}_Y_grid'] = y_range[np.clip(grid_indices[:, 1], 0, len(y_range) - 1)]
        indices_df[f'{zz}_T_grid'] = t_range[np.clip(grid_indices[:, 2], 0, len(t_range) - 1)]

    return df_grid, indices_df


# ------------------------------ #
#         MAIN WORKFLOW          #
# ------------------------------ #

def create_data_indices_all(data_source: str, keep_nanapis: bool, threshold: float):
    """
    Build data/production index maps basin-by-basin, then merge across basins.

    Parameters
    ----------
    data_source   : 'usgs' or 'ghgrp'
    keep_nanapis  : bool (USGS only)
    threshold     : non-hydrocarbon % sum threshold (USGS only)

    Returns
    -------
    data_indices_all     : DataFrame
    all_prod_indices_all : DataFrame
    """
    # ---- Load composition source ----
    if data_source == 'usgs':
        if keep_nanapis:
            data = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_with_nanapis.csv'))
        else:
            data = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_without_nanapis.csv'))

        def filter_non_hydrocarbons(df, thr):
            non_hc = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']
            df['non_hydrocarbon_sum'] = df[non_hc].fillna(0).sum(axis=1)
            out = df[df['non_hydrocarbon_sum'] < thr].drop(columns=['non_hydrocarbon_sum'])
            return out

        data = filter_non_hydrocarbons(data, threshold)
    else:
        data = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))

    # ---- Grid settings (space rescaled by alpha) ----
    alpha = 6.34
    res_space = 2000 / alpha
    res_time = 1825  # 5 years

    grid_params_dic = {}
    data_indices_list = []
    prod_indices_list = []
    prod_oil_indices_list = []
    components_seen = set()  # <- track all components encountered across basins

    basins_all = data.BASIN_NAME.unique()

    with tqdm(total=len(basins_all), desc="Basins processed") as pbar:
        for basin in basins_all:
            basin_file_path = os.path.join(root_path, f'wells_info_prod_per_basin/{basin}_final.csv')
            if os.path.exists(basin_file_path) and os.path.getsize(basin_file_path) > 0:
                # Composition (rescaled XY)
                df_data = data[data.BASIN_NAME == basin].reset_index(drop=True).copy()
                df_data['X'] = df_data['X'] / alpha
                df_data['Y'] = df_data['Y'] / alpha

                # Production (per-basin)
                wells_data = pd.read_csv(basin_file_path)
                df_prod = wells_data.reset_index(drop=True)
                df_prod = df_prod[df_prod['Monthly Gas'] > 0].reset_index(drop=True)
                if not df_prod.empty:
                    df_prod['X'] = df_prod['X'] / alpha
                    df_prod['Y'] = df_prod['Y'] / alpha
                    df_prod = df_prod[~pd.isna(df_prod['Monthly Gas'])].reset_index(drop=True)

                    # Build grid from production extents
                    _, grid_params = create_grid(df_prod, res_space, res_time)
                    grid_params_dic[basin] = grid_params

                    # Components present in this basin
                    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2',
                                  'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
                    components = [c for c in components if c in df_data.columns]
                    components_seen.update(components)

                    # Aggregate: composition (mean), production (sum)
                    data_on_grid, data_indices = aggregate(df_data, components, grid_params, mean_agg=True)
                    prod_on_grid, prod_indices = aggregate(df_prod, ['Monthly Gas'], grid_params, mean_agg=False)
                    prod_oil_on_grid, prod_oil_indices = aggregate(df_prod, ['Monthly Oil'], grid_params, mean_agg=False)

                    # Keep only cells with Gas > 0 (align shapes)
                    mask = prod_on_grid['Monthly Gas'] > 0
                    data_on_grid = data_on_grid[mask].reset_index(drop=True)
                    prod_oil_on_grid = prod_oil_on_grid[mask].reset_index(drop=True)
                    prod_on_grid = prod_on_grid[mask].reset_index(drop=True)

                    # Tag basin
                    data_indices['BASIN_NAME'] = basin
                    prod_indices['BASIN_NAME'] = basin
                    prod_oil_indices['BASIN_NAME'] = basin

                    data_indices_list.append(data_indices)
                    prod_indices_list.append(prod_indices)
                    prod_oil_indices_list.append(prod_oil_indices)

            pbar.update(1)

    # ---- Concatenate across basins (guard if lists are empty) ----
    if len(data_indices_list) == 0 or len(prod_indices_list) == 0 or len(prod_oil_indices_list) == 0:
        raise RuntimeError("No indices were generated. Check input paths and basin files.")

    data_indices_all = pd.concat(data_indices_list, ignore_index=True)
    prod_indices_all = pd.concat(prod_indices_list, ignore_index=True)
    prod_oil_indices_all = pd.concat(prod_oil_indices_list, ignore_index=True)

    # ---- Rescale coordinates back to original meters ----
    print('Rescaling coordinates...')
    for df_ in (data_indices_all, prod_indices_all, prod_oil_indices_all):
        df_['X'] = df_['X'] * alpha
        df_['Y'] = df_['Y'] * alpha

    # Per-variable voxel center rescale (use union of components across basins)
    for comp in sorted(components_seen):
        if f'{comp}_X_grid' in data_indices_all.columns:
            data_indices_all[f'{comp}_X_grid'] *= alpha
        if f'{comp}_Y_grid' in data_indices_all.columns:
            data_indices_all[f'{comp}_Y_grid'] *= alpha

    # Production voxel center rescale
    if 'Monthly Gas_X_grid' in prod_indices_all.columns:
        prod_indices_all['Monthly Gas_X_grid'] *= alpha
        prod_indices_all['Monthly Gas_Y_grid'] *= alpha
    if 'Monthly Oil_X_grid' in prod_oil_indices_all.columns:
        prod_oil_indices_all['Monthly Oil_X_grid'] *= alpha
        prod_oil_indices_all['Monthly Oil_Y_grid'] *= alpha

    # ---- Rename production index/centers to unified names ----
    print('Renaming columns...')
    prod_indices_all = prod_indices_all.rename(columns={
        'Monthly Gas_grid_index': 'Prod_index',
        'Monthly Gas_X_grid'   : 'Prod_X_grid',
        'Monthly Gas_Y_grid'   : 'Prod_Y_grid',
        'Monthly Gas_T_grid'   : 'Prod_T_grid'
    })

    prod_oil_indices_all = prod_oil_indices_all.rename(columns={
        'Monthly Oil_grid_index': 'Prod_index',
        'Monthly Oil_X_grid'    : 'Prod_X_grid',
        'Monthly Oil_Y_grid'    : 'Prod_Y_grid',
        'Monthly Oil_T_grid'    : 'Prod_T_grid'
    })

    # ---- Group/sum duplicates for Oil & Gas separately ----
    print('Grouping and summing oil ...')
    oil_group_cols = ['X','Y','T','Prod_index','Prod_X_grid','Prod_Y_grid','Prod_T_grid','BASIN_NAME']
    summed_duplicates_oil = prod_oil_indices_all.groupby(oil_group_cols, as_index=False)[['Monthly Oil']].sum()

    print('Grouping and summing gas ...')
    gas_group_cols = ['X','Y','T','Prod_index','Prod_X_grid','Prod_Y_grid','Prod_T_grid','BASIN_NAME']
    summed_duplicates_gas = prod_indices_all.groupby(gas_group_cols, as_index=False)[['Monthly Gas']].sum()

    print('Merging oil and gas ...')
    all_prod_indices_all = pd.merge(
        summed_duplicates_oil, summed_duplicates_gas,
        on=['X','Y','T','Prod_index','Prod_X_grid','Prod_Y_grid','Prod_T_grid','BASIN_NAME'],
        how='outer'
    )

    return data_indices_all, all_prod_indices_all


# ------------------------------ #
#              RUN               #
# ------------------------------ #

if __name__ == "__main__":
    data_source = 'usgs'   # or 'ghgrp'
    keep_nanapis = True
    threshold = 1000

    data_indices_all, all_prod_indices_all = create_data_indices_all(data_source, keep_nanapis, threshold)
    print("Done. Shapes:", data_indices_all.shape, all_prod_indices_all.shape)
