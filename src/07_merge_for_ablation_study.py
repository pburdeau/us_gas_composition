#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge per-basin prediction outputs with production-voxel indices,
optionally filter by production consistency, and export evaluation tables.

What this script does
---------------------
1) Load result CSVs & per-basin production voxel indices.
2) Align coordinates (rounding, rescaling) and merge on [X,Y,T].
3) Build test/validation sets per basin and compare true vs predicted values:
   - Kriging-refined results (result_all_new)
   - Simple kriging results (result_all)
   - NN results (result_all_nn)
4) Export merged/evaluation CSVs for C1 (and C2 for USGS), per (a,b) block split.

"""

# ------------------------------ #
#             SETUP              #
# ------------------------------ #

import warnings
warnings.filterwarnings('ignore')

import os
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.neighbors import KDTree
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.spatial import cKDTree  # kept as in your original

# ------------------------------ #
#           PATHS/IO             #
# ------------------------------ #

root_path = '/scratch/users/pburdeau/data/gas_composition'
ghgrp_path = os.sep.join([root_path, 'ghgrp', 'ghgrp_v2.csv'])
ghgrp = pd.read_csv(ghgrp_path)
shapefiles_path = root_path + '/shapefiles'
out_path = root_path + '/out'
os.makedirs(out_path, exist_ok=True)  # ensure output folder exists


# ------------------------------ #
#         MERGE HELPERS          #
# ------------------------------ #

def upload_result(all_prod_indices_all: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Load result CSVs for a glob 'path' under out/, merge them with production-voxel indices
    for each basin on rounded coordinates and T, and return a single merged DataFrame.

    Parameters
    ----------
    all_prod_indices_all : DataFrame
        Contains ['BASIN_NAME','X','Y','T','Prod_X_grid','Prod_Y_grid','Prod_T_grid', ...]
    path : str
        Glob pattern for results, relative to root_path/out/

    Returns
    -------
    DataFrame
        Merged table including ['X_prod','Y_prod','T_prod'] from the grid alignment.
    """
    csv_files_pattern = os.path.join(root_path, 'out', path)
    csv_files = glob.glob(csv_files_pattern)

    if not csv_files:
        raise FileNotFoundError(f"No files matched pattern: {csv_files_pattern}")

    result_all_list = [pd.read_csv(file) for file in csv_files]
    result_all = pd.concat(result_all_list, ignore_index=True)

    # Align (X,Y) with production grid rounding convention
    result_all['X'] = (result_all['X'] * 6.34).round(1)
    result_all['Y'] = (result_all['Y'] * 6.34).round(1)

    merged_list = []
    basins = result_all.BASIN_NAME.unique()

    with tqdm(total=len(basins)) as pbar:
        for basin in basins:
            basin_result = result_all[result_all.BASIN_NAME == basin].copy()
            basin_prod = all_prod_indices_all[all_prod_indices_all.BASIN_NAME == basin].copy()

            if basin_result.empty or basin_prod.empty:
                pbar.update(1)
                continue

            # Round production voxel coords to match
            basin_prod['Prod_X_grid'] = basin_prod['Prod_X_grid'].round(1)
            basin_prod['Prod_Y_grid'] = basin_prod['Prod_Y_grid'].round(1)

            # Merge on rounded (X,Y) and exact T
            merged = pd.merge(
                basin_prod,
                basin_result,
                left_on=['Prod_X_grid', 'Prod_Y_grid', 'Prod_T_grid'],
                right_on=['X', 'Y', 'T'],
                suffixes=('', '_result')
            )

            # Cleanup & rename (preserve your schema)
            merged = merged.drop(columns=['X_result', 'Y_result', 'T_result', 'BASIN_NAME_result'], errors='ignore')
            merged = merged.rename(columns={'X': 'X_prod', 'Y': 'Y_prod', 'T': 'T_prod'})

            merged_list.append(merged)
            pbar.update(1)

    if not merged_list:
        # Return empty frame with a reasonable schema if nothing merged
        return pd.DataFrame(columns=list(all_prod_indices_all.columns) + list(result_all.columns))

    final_merged = pd.concat(merged_list, ignore_index=True)
    return final_merged


def filter_non_hydrocarbons(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Drop rows where non-hydrocarbon sum exceeds threshold.
    """
    non_hydrocarbons = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']
    df['non_hydrocarbon_sum'] = df[non_hydrocarbons].fillna(0).sum(axis=1)
    filtered_df = df[df['non_hydrocarbon_sum'] < threshold].drop(columns=['non_hydrocarbon_sum'])
    return filtered_df


def filter_threshold_production(data_source: str,
                                wells_data: pd.DataFrame,
                                other_data: pd.DataFrame,
                                REL_DIFF_THRESHOLD: float) -> pd.DataFrame:
    """
    Keep only basins whose mean Monthly Gas & Oil are consistent between wells_data and other_data
    within REL_DIFF_THRESHOLD (relative difference).

    For GHGRP, restrict wells_data to Year>=2015.
    """
    if data_source == 'ghgrp':
        wells_data = wells_data[wells_data.Year >= 2015]

    gas_by_basin_1 = wells_data.groupby('BASIN_NAME')['Monthly Gas'].mean()
    gas_by_basin_2 = other_data.groupby('BASIN_NAME')['Monthly Gas'].mean()

    oil_by_basin_1 = wells_data.groupby('BASIN_NAME')['Monthly Oil'].mean()
    oil_by_basin_2 = other_data.groupby('BASIN_NAME')['Monthly Oil'].mean()

    gas_comparison = pd.DataFrame({'gas_1': gas_by_basin_1, 'gas_2': gas_by_basin_2}).dropna()
    oil_comparison = pd.DataFrame({'oil_1': oil_by_basin_1, 'oil_2': oil_by_basin_2}).dropna()

    gas_comparison['rel_diff_gas'] = abs(gas_comparison['gas_1'] - gas_comparison['gas_2']) / gas_comparison['gas_1']
    oil_comparison['rel_diff_oil'] = abs(oil_comparison['oil_1'] - oil_comparison['oil_2']) / oil_comparison['oil_1']

    valid_gas_basins = gas_comparison[gas_comparison['rel_diff_gas'] <= REL_DIFF_THRESHOLD].index
    valid_oil_basins = oil_comparison[oil_comparison['rel_diff_oil'] <= REL_DIFF_THRESHOLD].index
    valid_basins = valid_gas_basins.intersection(valid_oil_basins)
    return other_data[other_data['BASIN_NAME'].isin(valid_basins)]


# ------------------------------ #
#     EVALUATION MERGE/COMPARE   #
# ------------------------------ #

def return_merged(test_block: pd.DataFrame,
                  df_test: pd.DataFrame,
                  result_all_new: pd.DataFrame,
                  result_all: pd.DataFrame,
                  result_all_nn: pd.DataFrame,
                  comp: str,
                  filter_prod: bool):
    """
    Build merged evaluation frames (kriging refined, simple kriging, nn) vs. the test block.

    Returns
    -------
    (merged_kriging, merged_simple_kriging, merged_nn) : tuple of DataFrames
    """

    def prepare_and_analyze(df1: pd.DataFrame, df2: pd.DataFrame, comp: str) -> pd.DataFrame:
        """
        For df1 (test rows) and df2 (predictions), compute nearest distances per basin,
        then merge and compute absolute/squared/normalized squared errors.
        """
        df1_copy = df1.copy()
        df2_copy = df2.copy()

        print('Merging ...', flush=True)
        df2_unique = df2_copy.merge(df1_copy,
                                    on=['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME'],
                                    how='left', indicator=True)
        df2_unique = df2_unique[df2_unique['_merge'] == 'left_only'].drop(columns=['_merge'])

        # Basin-by-basin nearest neighbor distances
        for basin in tqdm(df1_copy['BASIN_NAME'].unique(), desc="Processing Basins"):
            df1_basin = df1_copy[df1_copy['BASIN_NAME'] == basin]
            df2_basin_unique = df2_unique[df2_unique['BASIN_NAME'] == basin]

            if not df2_basin_unique.empty:
                tree = KDTree(df2_basin_unique[['X_prod', 'Y_prod', 'T_prod']])
                closest_distances = tree.query(df1_basin[['X_prod', 'Y_prod', 'T_prod']].to_numpy())[0]
                df1_copy.loc[df1_basin.index, 'closest_distance'] = closest_distances

        # Compare to model (inner-join on the same keys)
        merged_basin_data = pd.merge(df1_copy, df2_copy,
                                     on=['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME'])
        merged_basin_data = merged_basin_data[~pd.isna(merged_basin_data[comp + '_x'])]
        merged_basin_data = merged_basin_data[~pd.isna(merged_basin_data[comp + '_y'])]

        # Error metrics
        merged_basin_data['abs_error'] = np.abs(merged_basin_data[comp + '_y'] - merged_basin_data[comp + '_x'])
        merged_basin_data['sq_error'] = (merged_basin_data[comp + '_y'] - merged_basin_data[comp + '_x']) ** 2
        merged_basin_data['nse'] = ((merged_basin_data[comp + '_y'] - merged_basin_data[comp + '_x']) /
                                    merged_basin_data['std_C1']) ** 2
        print('Merged.', flush=True)
        return merged_basin_data

    # Right-merge to keep all test block rows; mark membership
    merged = df_test.merge(test_block[['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME']],
                           on=['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME'],
                           how='right',
                           indicator=True)

    # Optional production-consistency filter
    if filter_prod:
        wells_data = []
        basin_list = merged.BASIN_NAME.unique()
        with tqdm(total=len(basin_list)) as pbar:
            for basin_name in basin_list:
                basin_file_path = os.path.join(root_path, f'wells_info_prod_per_basin/{basin_name}_final.csv')
                if os.path.isfile(basin_file_path):
                    wells_data.append(pd.read_csv(basin_file_path))
                pbar.update(1)
        if wells_data:  # only filter if we actually found files
            wells_data = pd.concat(wells_data, ignore_index=True)
            merged = filter_threshold_production(data_source, wells_data, merged, 0.9)

    # Fill NA for component columns on rows not present in df_test
    cols_to_null = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR',
                    'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
    merged.loc[merged['_merge'] == 'left_only', cols_to_null] = pd.NA

    merged.drop(columns=['_merge'], inplace=True)
    merged = merged.drop_duplicates(subset=['X_prod', 'Y_prod', 'T_prod'])

    # df1: test rows to compare against predictions
    df1 = merged.reset_index(drop=True)

    # Prepare three prediction sources and clean small artifacts
    df2_kriging = result_all_new.reset_index(drop=True)
    df2_kriging = df2_kriging.drop_duplicates(subset=['X_prod', 'Y_prod', 'T_prod'])
    df2_kriging = df2_kriging.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'}, errors='ignore')

    df2_simple_kriging = result_all[~pd.isna(result_all[comp + '_predic_gor'])].reset_index(drop=True)
    df2_simple_kriging = df2_simple_kriging.drop_duplicates(subset=['X_prod', 'Y_prod', 'T_prod'])
    df2_simple_kriging = df2_simple_kriging.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'}, errors='ignore')

    df2_nn = result_all_nn[~pd.isna(result_all_nn[comp + '_predic_gor'])].reset_index(drop=True)
    df2_nn = df2_nn.drop_duplicates(subset=['X_prod', 'Y_prod', 'T_prod'])
    df2_nn = df2_nn.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'}, errors='ignore')

    # Build all three merged evaluation frames
    merged_kriging = prepare_and_analyze(df1.copy(), df2_kriging, comp)
    merged_simple_kriging = prepare_and_analyze(df1.copy(), df2_simple_kriging, comp)
    merged_nn = prepare_and_analyze(df1.copy(), df2_nn, comp)
    return merged_kriging, merged_simple_kriging, merged_nn


# ------------------------------ #
#         MERGE DRIVER           #
# ------------------------------ #

def merge_results(threshold: int,
                  data_source: str,
                  keep_nanapis: bool,
                  radius_factor: float,
                  filter_prod: bool):
    """
    For each (a,b) block split:
      - Load result CSVs and block indices.
      - Build df_data and df_test (with *_prod columns only where needed).
      - Create merged evaluation frames and export them.
    """
    # Load production indices (prebuilt tables)
    if data_source == 'ghgrp':
        all_prod_indices_all = pd.read_csv('/scratch/users/pburdeau/notebooks/all_prod_indices_all_ghgrp.csv')
    else:
        all_prod_indices_all = pd.read_csv(f'/scratch/users/pburdeau/notebooks/all_prod_indices_all_{threshold}.csv')

    # Match rounding convention
    all_prod_indices_all['X'] = (all_prod_indices_all['X']).round(1)
    all_prod_indices_all['Y'] = (all_prod_indices_all['Y']).round(1)
    all_prod_indices_all['Prod_X_grid'] = all_prod_indices_all['Prod_X_grid'].round(1)
    all_prod_indices_all['Prod_Y_grid'] = all_prod_indices_all['Prod_Y_grid'].round(1)

    with tqdm(total=len([-1, 1, -1, 1])) as pbar:
        for a, b in zip([-1, 1, -1, 1], [-1, 1, 1, -1]):
            a = str(a); b = str(b)
            print(f'Uploading results...', flush=True)

            # Load 3 result sources for this (a,b)
            result_all_nn = pd.read_csv(
                os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_all_nn_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv')
            )
            print(f'Uploaded result_all_nn.', flush=True)

            result_all_new = pd.read_csv(
                os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_all_new_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv')
            )
            print(f'Uploaded result_all_new.', flush=True)

            result_all = pd.read_csv(
                os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_all_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv')
            )
            print(f'Uploaded result_all.', flush=True)

            # Test block for this (a,b)
            test_block = upload_result(
                all_prod_indices_all,
                f'block_test_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_sample_0_a_{a}_b_{b}_basin_*_radius_factor_{radius_factor}.csv'
            )

            # Base data source table; set *_prod ONLY for GHGRP (your original logic)
            if data_source == 'usgs':
                df_data = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_prod.csv'))
            else:
                df_data = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))
                df_data['X_prod'] = df_data['X']
                df_data['Y_prod'] = df_data['Y']
                df_data['T_prod'] = df_data['T']

            if data_source == 'usgs':
                df_data = filter_non_hydrocarbons(df_data, threshold)

            # Rounding for coordinate keys
            df_data['X_prod'] = (df_data['X_prod']).round(1)
            df_data['Y_prod'] = (df_data['Y_prod']).round(1)

            # Build test table via inner-join on *_prod coordinates
            df_test = test_block[['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME']].merge(
                df_data, on=['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME']
            )

            if data_source == 'usgs':
                # USGS optional filtering
                df_test = filter_non_hydrocarbons(df_test, threshold)

                # Merge/evaluate for C2
                merged_kriging_C2, merged_simple_kriging_C2, merged_nn_C2 = return_merged(
                    test_block, df_test, result_all_new, result_all, result_all_nn, 'C2', filter_prod
                )

                print('Finished merging C2, adding merged_simple_kriging_C2...', flush=True)
                merged_simple_kriging_C2.to_csv(
                    os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_simple_kriging_C2_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'),
                    index=False
                )

                del merged_simple_kriging_C2
                print('Added merged_simple_kriging_C2, adding merged_kriging_C2...', flush=True)

                merged_kriging_C2.to_csv(
                    os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_kriging_C2_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'),
                    index=False
                )

                del merged_kriging_C2
                print('Added merged_kriging_C2, adding merged_nn_C2...', flush=True)

                merged_nn_C2.to_csv(
                    os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_nn_C2_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'),
                    index=False
                )

                del merged_nn_C2
                print('Added merged_nn_C2, merging C1...', flush=True)

                # Merge/evaluate for C1
                merged_kriging_C1, merged_simple_kriging_C1, merged_nn_C1 = return_merged(
                    test_block, df_test, result_all_new, result_all, result_all_nn, 'C1', filter_prod
                )

                # Clear large frames to free memory before writing
                del result_all, result_all_new, result_all_nn, test_block, df_test

                print('Finished merging C1, adding merged_simple_kriging_C1...', flush=True)
                merged_simple_kriging_C1.to_csv(
                    os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_simple_kriging_C1_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'),
                    index=False
                )

                del merged_simple_kriging_C1
                print('Added merged_simple_kriging_C1, adding merged_kriging_C1...', flush=True)

                merged_kriging_C1.to_csv(
                    os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_kriging_C1_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'),
                    index=False
                )

                del merged_kriging_C1
                print('Added merged_kriging_C1, adding merged_nn_C1...', flush=True)

                merged_nn_C1.to_csv(
                    os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_nn_C1_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'),
                    index=False
                )

                del merged_nn_C1
                print(f'Added merged_nn_C1, finished a = {a}, b = {b}', flush=True)
                pbar.update(1)

            else:
                # GHGRP: only C1
                merged_kriging_C1, merged_simple_kriging_C1, merged_nn_C1 = return_merged(
                    test_block, df_test, result_all_new, result_all, result_all_nn, 'C1', filter_prod
                )

                del result_all, result_all_new, result_all_nn, test_block, df_test

                print('Finished merging C1, adding merged_simple_kriging_C1...', flush=True)
                merged_simple_kriging_C1.to_csv(
                    os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_simple_kriging_C1_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'),
                    index=False
                )

                del merged_simple_kriging_C1
                print('Added merged_simple_kriging_C1, adding merged_kriging_C1...', flush=True)

                merged_kriging_C1.to_csv(
                    os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_kriging_C1_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'),
                    index=False
                )

                del merged_kriging_C1
                print('Added merged_kriging_C1, adding merged_nn_C1...', flush=True)

                merged_nn_C1.to_csv(
                    os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_merged_nn_C1_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'),
                    index=False
                )

                del merged_nn_C1
                print(f'Added merged_nn_C1, finished a = {a}, b = {b}', flush=True)
                pbar.update(1)

    return None


# ------------------------------ #
#               CLI              #
# ------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=int)
    parser.add_argument('--radius_factor', type=int)
    parser.add_argument('--filter_prod', type=int)

    arguments = parser.parse_args()

    data_sources = ['usgs', 'ghgrp']
    data_source = data_sources[arguments.data_source]

    radius_factors = [0.15, 0.175]
    radius_factor = radius_factors[arguments.radius_factor]

    filter_prods = [True, False]
    filter_prod = filter_prods[arguments.filter_prod]

    keep_nanapis = True
    threshold = 1000

    merge_results(threshold, data_source, keep_nanapis, radius_factor, filter_prod)
