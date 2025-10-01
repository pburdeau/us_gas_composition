#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate non-aggregated results into per-grid summaries, save/load result dictionaries,
select rows closest to target years per (X,Y,BASIN,component), propagate uncertainties
for non-(C1,CO2) renormalization, and export Leaflet-ready CSVs per basin/component.
"""

# =========================
# Imports & Global Config
# =========================
import os
import pickle
import numpy as np
import pandas as pd

from tqdm import tqdm

# -----------------
# Paths (config)
# -----------------
root_path = '/scratch/users/pburdeau/data/gas_composition'
ghgrp_path = os.sep.join([root_path, 'ghgrp', 'ghgrp_v2.csv'])
ghgrp = pd.read_csv(ghgrp_path)
shapefiles_path = root_path + '/shapefiles'
out_path = root_path + '/for_download'
os.makedirs(out_path, exist_ok=True)


# =========================================
# Aggregate + save non-aggregated results
# =========================================
def save_results(data_source: str):
    """
    Read non-aggregated results from for_download (out_path), compute
    weighted averages by grid cell (weights = Monthly Gas), and store
    aggregated DataFrames in a dict keyed by (threshold, a, b).
    Finally, pickle the dict as "new_result_dic_r0175_{data_source}.pkl".
    """
    if data_source == 'ghgrp':
        thresholds = [1000]
        comp_columns = ['CO2','C1', 'std_CO2', 'C1_predic_gor', 'C1_predic_kriging', 'std_C1']
        first_value_columns = ['beta_C1', 'BASIN_NAME']
        radius_factor = 0.15
    else:
        thresholds = [1000]
        comp_columns = [
            'HE','CO2','H2','N2','H2S','AR','O2','C1','C2',
            'C3','N-C4','I-C4','N-C5','I-C5','C6+',
            'std_HE','std_CO2','C1_predic_gor','C1_predic_kriging',
            'C2_predic_gor','C2_predic_kriging','std_H2','std_N2','std_H2S',
            'std_AR','std_O2','std_C1','std_C2','std_C3','std_N-C4','std_I-C4',
            'std_N-C5','std_I-C5','std_C6+'
        ]
        first_value_columns = ['beta_C1', 'beta_C2', 'BASIN_NAME']
        radius_factor = 0.175

    result_dic = {}
    combinations = list(zip([1], [1]))  # (a,b) = (1,1) only, per your code

    sum_columns = ['Monthly Gas', 'Monthly Oil']
    group_cols = ['Prod_X_grid', 'Prod_Y_grid', 'Prod_T_grid']

    with tqdm(total=len(combinations) * len(thresholds), desc=f"Aggregating {data_source}") as pbar:
        for (a, b) in combinations:
            for threshold in thresholds:
                keep_nanapis = True
                sample = 2
                # Read from your for_download directory
                file_path = os.path.join(
                    out_path,
                    f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_sample_{sample}_a_{a}_b_{b}_radius_factor_{radius_factor}.csv'
                )

                df = pd.read_csv(file_path)
                # Drop helper columns if present (no error if absent)
                df.drop(columns=['Unnamed: 0', 'X_prod', 'Y_prod', 'T_prod', 'check_sum'], errors='ignore', inplace=True)

                # Round production grid coords (merge convention)
                df['Prod_X_grid'] = df['Prod_X_grid'].round(1)
                df['Prod_Y_grid'] = df['Prod_Y_grid'].round(1)

                # Weighted averages with Monthly Gas
                weighted_avgs = df[group_cols + comp_columns].copy()
                weighted_avgs[comp_columns] = weighted_avgs[comp_columns].multiply(df['Monthly Gas'], axis=0)
                weighted_avgs = weighted_avgs.groupby(group_cols).sum(min_count=1)

                total_weights = df.groupby(group_cols)['Monthly Gas'].sum(min_count=1)
                total_weights.replace(0, np.nan, inplace=True)
                weighted_avgs = weighted_avgs.div(total_weights, axis=0)

                # Sums + first-value columns per cell
                agg_funcs = {col: 'sum' for col in sum_columns}
                agg_funcs.update({col: 'first' for col in first_value_columns})

                df_grouped = df.groupby(group_cols).agg(agg_funcs).reset_index()
                df_grouped = df_grouped.merge(weighted_avgs, on=group_cols, how='left')

                # Rename grid keys to X,Y,T and add Year using your reference epoch
                df_grouped = df_grouped.rename(columns={
                    'Prod_X_grid':'X','Prod_Y_grid':'Y','Prod_T_grid':'T'
                })
                reference_date = '1918-08-19T00:00:00.000000000'
                df_grouped['Year'] = (pd.Timestamp(reference_date) + pd.to_timedelta(df_grouped['T'], unit='D')).dt.year

                # Save into the dict
                result_dic[(threshold, a, b)] = df_grouped
                pbar.update(1)

    # Pickle to new_result_dic_r0175_{data_source}.pkl (kept your name)
    with open(f"new_result_dic_r0175_{data_source}.pkl", "wb") as f:
        pickle.dump(result_dic, f)

    return result_dic


# -------------------------
# Build result dictionaries
# -------------------------
result_dic_usgs = save_results('usgs')
result_dic_ghgrp = save_results('ghgrp')


# =========================================
# Select rows closest to target years
# =========================================
ghgrp_production_gdf = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))

def filter_to_closest_years_by_xy_and_component(df: pd.DataFrame,
                                                components: list[str],
                                                year_dict_all: dict[str, dict]) -> pd.DataFrame:
    """
    For each component, keep the row for each (X, Y, BASIN_NAME) whose Year is closest to
    (target_year - 5) but no more than 5 years away, using (Year + 5 - target).abs().
    Then assemble a wide frame with one row per cell and columns comp & std_comp.
    """
    cells = df[['X','Y','BASIN_NAME']].drop_duplicates().reset_index(drop=True)
    result = cells.copy()

    for comp in tqdm(components, desc="Components"):
        std_col = f"std_{comp}"
        if comp not in df.columns:
            result[comp] = np.nan
            result[std_col] = np.nan
            continue

        df_comp = df[df[comp].notna()].copy()
        year_dict = year_dict_all.get(comp, {})

        def get_closest_idx(group):
            basin = group['BASIN_NAME'].iat[0]
            if basin not in year_dict:
                return None
            target = year_dict[basin]
            adjusted_diff = (group['Year'] + 5 - target).abs()
            min_diff = adjusted_diff.min()
            if min_diff <= 5:
                return adjusted_diff.idxmin()
            return None

        closest_indices = (
            df_comp
            .groupby(['X','Y','BASIN_NAME'])
            .apply(get_closest_idx)
            .dropna()
            .astype(int)
        )
        df_filt = df_comp.loc[closest_indices.values, ['X','Y','BASIN_NAME', comp, std_col]]
        result = result.merge(df_filt, on=['X','Y','BASIN_NAME'], how='left')

    return result


# -------------------------
# Load dicts & build slices
# -------------------------
data_source = 'ghgrp'
with open(f"new_result_dic_r0175_{data_source}.pkl", "rb") as f:
    result_dic = pickle.load(f)

radius_factor = 0.15
threshold = 1000
a = 1
b = 1
result_ghgrp = result_dic[(threshold, a, b)]

components = ['C1', 'CO2']
year_ghgrp_dict_all = {
    comp: ghgrp_production_gdf[ghgrp_production_gdf[comp].notna()]
            .groupby('BASIN_NAME')['Year'].max().to_dict()
    for comp in components
}
result_ghgrp_filtered = filter_to_closest_years_by_xy_and_component(
    result_ghgrp, ['C1', 'CO2'], year_ghgrp_dict_all
)

data_source = 'usgs'
with open(f"new_result_dic_r0175_{data_source}.pkl", "rb") as f:
    result_dic = pickle.load(f)

sample = 2
radius_factor = 0.175
result_usgs = result_dic[(threshold, a, b)]


# =========================================
# USGS raw data & per-component latest year
# =========================================
def create_data_structure():
    """
    Return a callable that loads & filters USGS data by a non-hydrocarbon threshold
    and whether to keep NaN-API rows.
    """
    def filter_dataframe(threshold, keep_nanapis):
        if keep_nanapis:
            df = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_with_nanapis.csv'))
        else:
            df = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_without_nanapis.csv'))

        non_hc = ['HE','CO2','H2','N2','H2S','AR','O2']
        df['non_hydrocarbon_sum'] = df[non_hc].sum(axis=1, skipna=True)
        filtered = df[df['non_hydrocarbon_sum'] < threshold].drop(columns=['non_hydrocarbon_sum'])
        return filtered
    return filter_dataframe

filter_func = create_data_structure()
raw_data_usgs = filter_func(1000, True).copy()

components_all = ['HE','CO2','H2','N2','H2S','AR','O2','C1','C2','C3','N-C4','I-C4','N-C5','I-C5','C6+']
year_usgs_dict_all = {
    comp: raw_data_usgs[raw_data_usgs[comp].notna()]
            .groupby('BASIN_NAME')['Year'].max().to_dict()
    for comp in components_all
}

result_usgs_filtered = filter_to_closest_years_by_xy_and_component(
    result_usgs, components_all, year_usgs_dict_all
)


# =========================================
# Renormalize non-(C1,CO2) + propagate σ
# =========================================
def propagate_uncertainties_normalization(data_on_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize non-C1/CO2 components to 100 - C1 - CO2 and propagate 1σ uncertainties.
    - If C1+CO2>100: other comps set to 0.
    - Avoid negative values; keep C1/CO2 unchanged except clipping to maintain sum ≤ 100.
    """
    comps_all = ['HE','CO2','H2','N2','H2S','AR','O2','C1','C2','C3','N-C4','I-C4','N-C5','I-C5','C6+']
    comps_fixed = ['C1','CO2']
    comps_renorm = [c for c in comps_all if c not in comps_fixed]

    A = data_on_grid[comps_renorm].to_numpy(dtype=float)
    σ = data_on_grid[['std_' + c for c in comps_renorm]].to_numpy(dtype=float)

    A_total = np.nansum(A, axis=1)
    C1 = data_on_grid['C1'].to_numpy(dtype=float)
    CO2 = data_on_grid['CO2'].to_numpy(dtype=float)
    se_C1 = data_on_grid.get('std_C1', pd.Series(np.nan, index=data_on_grid.index)).to_numpy(dtype=float)
    se_CO2 = data_on_grid.get('std_CO2', pd.Series(np.nan, index=data_on_grid.index)).to_numpy(dtype=float)

    sum_C1_CO2 = C1 + CO2
    sum_C1_CO2_clipped = np.minimum(sum_C1_CO2, 100)
    rem = 100 - sum_C1_CO2_clipped

    F = np.zeros_like(A)
    abs_err = np.zeros_like(σ)

    valid = (rem > 0) & (A_total > 0)
    if np.any(valid):
        scale = (rem[valid] / A_total[valid])[:, None]
        F_valid = np.clip(A[valid] * scale, 0, None)
        abs_err_valid = σ[valid] * scale
        F[valid] = F_valid
        abs_err[valid] = abs_err_valid

    for i, c in enumerate(comps_renorm):
        data_on_grid[c] = F[:, i]
        data_on_grid['std_' + c] = abs_err[:, i]

    C1_clipped = np.clip(C1, 0, 100)
    CO2_clipped = np.clip(CO2, 0, 100 - C1_clipped)
    data_on_grid['C1'] = C1_clipped
    data_on_grid['std_C1'] = se_C1
    data_on_grid['CO2'] = CO2_clipped
    data_on_grid['std_CO2'] = se_CO2

    data_on_grid['sum_check'] = data_on_grid[comps_all].sum(axis=1).round(5)
    return data_on_grid


# =========================================
# Merge GHGRP subset into USGS grid & export
# =========================================
ghgrp_subset = result_ghgrp_filtered[['X','Y','BASIN_NAME','C1','std_C1','CO2','std_CO2']].copy()

combined_df = result_usgs_filtered.copy().drop(
    columns=['C1','std_C1','CO2','std_CO2'], errors='ignore'
)
combined_df = combined_df.merge(ghgrp_subset, on=['X','Y','BASIN_NAME'], how='left')

components_export = [
    'C1','CO2','HE','H2','N2','H2S','AR','O2','C2','C3','N-C4','I-C4','N-C5','I-C5','C6+'
]

basins = combined_df['BASIN_NAME'].dropna().unique()
output_dir = "datasets_leaflet_new"
os.makedirs(output_dir, exist_ok=True)

for basin in basins:
    basin_sanitized = basin.replace(" ", "_").replace("/", "_")
    basin_dir = os.path.join(output_dir, basin_sanitized)
    os.makedirs(basin_dir, exist_ok=True)

    df_basin = combined_df[combined_df['BASIN_NAME'] == basin].copy()

    for comp in components_export:
        for prefix in ['', 'std_']:
            col = prefix + comp
            if col not in df_basin.columns:
                continue
            df_sub = df_basin[['X','Y',col]].dropna().copy()
            df_sub = df_sub.rename(columns={col: 'value'})
            out_csv = os.path.join(basin_dir, f"dataset_{col}.csv")
            df_sub.to_csv(out_csv, index=False)
