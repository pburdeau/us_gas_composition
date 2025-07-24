root_path = '/scratch/users/pburdeau/data/gas_composition'
ghgrp_path = os.sep.join([root_path, 'ghgrp', 'ghgrp_v2.csv'])
ghgrp = pd.read_csv(ghgrp_path)
shapefiles_path = root_path + '/shapefiles'
out_path = root_path + '/out'
out_path = root_path + '/for_download'

# upload the non-aggregated results, aggregate them, and create a dict to save them

def save_results(data_source):
    if data_source == 'ghgrp':
        thresholds = [1000]
        comp_columns = ['CO2','C1', 'std_CO2', 'C1_predic_gor', 'C1_predic_kriging', 'std_C1']
        first_value_columns = ['beta_C1', 'BASIN_NAME']
        radius_factor = 0.15
    else:
#         thresholds = [1000]

        thresholds = [1000]
        comp_columns = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2',
                'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+', 'std_HE', 'std_CO2',
                'C1_predic_gor', 'C1_predic_kriging',
                'C2_predic_gor', 'C2_predic_kriging', 'std_H2', 'std_N2', 'std_H2S',
                'std_AR', 'std_O2', 'std_C1', 'std_C2', 'std_C3', 'std_N-C4', 'std_I-C4',
                'std_N-C5', 'std_I-C5', 'std_C6+']
        first_value_columns = ['beta_C1', 'beta_C2', 'BASIN_NAME']
        radius_factor = 0.175

    result_dic = {}
#     combinations = list(zip([-1, 1, -1, 1], [-1, 1, 1, -1]))
    combinations = list(zip([1], [1]))

    sum_columns = ['Monthly Gas', 'Monthly Oil']
    group_cols = ['Prod_X_grid', 'Prod_Y_grid', 'Prod_T_grid']

    with tqdm(total=len(combinations) * len(thresholds)) as pbar:
        for (a, b) in combinations:
            for threshold in thresholds:
                keep_nanapis = True
                sample = 2
                file_path = os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_sample_{sample}_a_{a}_b_{b}_radius_factor_{radius_factor}.csv')
#                 download_dir = os.path.join(root_path, 'for_download')
#                 os.makedirs(download_dir, exist_ok=True)

                
                
                df = pd.read_csv(file_path)
                print('Read df')
                df.drop(columns=['Unnamed: 0', 'X_prod', 'Y_prod', 'T_prod', 'check_sum'], errors='ignore', inplace=True)
                print('Aggregating...')
                df['Prod_X_grid'] = df['Prod_X_grid'].round(1)
                df['Prod_Y_grid'] = df['Prod_Y_grid'].round(1)
                weighted_avgs = df[group_cols + comp_columns].copy()
                weighted_avgs[comp_columns] = weighted_avgs[comp_columns].multiply(df['Monthly Gas'], axis=0)
                weighted_avgs = weighted_avgs.groupby(group_cols).sum(min_count=1)
                total_weights = df.groupby(group_cols)['Monthly Gas'].sum(min_count=1)
                total_weights.replace(0, np.nan, inplace=True)
                weighted_avgs = weighted_avgs.div(total_weights, axis=0)

                agg_funcs = {col: 'sum' for col in sum_columns}
                agg_funcs.update({col: 'first' for col in first_value_columns})

                df_grouped = df.groupby(group_cols).agg(agg_funcs).reset_index()
                df_grouped = df_grouped.merge(weighted_avgs, on=group_cols, how='left')

                print('Aggregated.')

                # Rename and convert to year
                df_grouped = df_grouped.rename(columns={'Prod_X_grid':'X','Prod_Y_grid':'Y','Prod_T_grid':'T'})
                reference_date = '1918-08-19T00:00:00.000000000'
                df_grouped['Year'] = (pd.Timestamp(reference_date) + pd.to_timedelta(df_grouped['T'], unit='D')).dt.year

                # Store in dictionary
                result_dic[(threshold, a, b)] = df_grouped

                pbar.update(1)

    with open(f"new_result_dic_r0175_{data_source}.pkl", "wb") as f:
        pickle.dump(result_dic, f)
    return result_dic

result_dic_usgs = save_results('usgs')
result_dic_usgs = save_results('ghgrp')

# Code to make data available!!
ghgrp_production_gdf = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))  # all basins

def filter_to_closest_years_by_xy_and_component(df, components, year_dict_all):
    """
    For each component, keep the row for each (X, Y, BASIN_NAME) whose Year
    is closest to (target_year - 5) but no more than 5 years away, using your
    `(Year + 5 - target).abs()` logic.  Then assemble a wide DataFrame
    with one row per cell and columns comp & std_comp.
    """
    # 1) master list of all cells
    cells = df[['X','Y','BASIN_NAME']].drop_duplicates().reset_index(drop=True)
    result = cells.copy()

    # 2) loop over components
    for comp in tqdm(components, desc="Components"):
        std_col = f"std_{comp}"
        # if this comp never even in df, just fill NaNs
        if comp not in df.columns:
            result[comp]   = np.nan
            result[std_col] = np.nan
            continue

        # filter to rows where comp exists
        df_comp = df[df[comp].notna()].copy()
        year_dict = year_dict_all[comp]

        def get_closest_idx(group):
            basin = group['BASIN_NAME'].iat[0]
            if basin not in year_dict:
                return None
            target = year_dict[basin]
            # your original logic:
            adjusted_diff = (group['Year'] + 5 - target).abs()
            min_diff = adjusted_diff.min()
            if min_diff <= 5:
                return adjusted_diff.idxmin()
            return None

        # one pick per (X,Y,BASIN_NAME)
        closest_indices = (
            df_comp
              .groupby(['X','Y','BASIN_NAME'])
              .apply(get_closest_idx)
              .dropna()
              .astype(int)
        )

        # extract those rows and only the columns we need
        df_filt = df_comp.loc[closest_indices.values, 
                              ['X','Y','BASIN_NAME', comp, std_col]]

        # merge in: result has no comp/std_col yet, so no suffix conflicts
        result = result.merge(
            df_filt,
            on=['X','Y','BASIN_NAME'],
            how='left'
        )

    return result

data_source = 'ghgrp'
with open(f"result_dic_r0175_{data_source}.pkl", "rb") as f:
    result_dic = pickle.load(f)

radius_factor = 0.15
threshold = 1000
a = 1
b = 1
result_ghgrp = result_dic[(threshold, a, b)]

components = ['C1', 'CO2']
year_ghgrp_dict_all = {
    comp: {k: v for k, v in ghgrp_production_gdf[ghgrp_production_gdf[comp].notna()]
           .groupby('BASIN_NAME')['Year'].max().to_dict().items()}
    for comp in components
}

result_ghgrp_filtered = filter_to_closest_years_by_xy_and_component(result_ghgrp, ['C1', 'CO2'], year_ghgrp_dict_all)

# result_ghgrp = result_ghgrp.drop(columns='std_CO2')
# result_ghgrp = result_ghgrp.rename(columns={'std_CO2_mean':'std_CO2'})

data_source = 'usgs'
with open(f"result_dic_r0175_{data_source}.pkl", "rb") as f:
    result_dic = pickle.load(f)

sample = 2
radius_factor = 0.175
result_usgs = result_dic[(threshold, a, b)]


# import raw data from usgs

def create_data_structure():
    def filter_dataframe(threshold, keep_nanapis):
        """
        Filters the usgs DataFrame based on threshold and keep_nanapis.
        """
        # Handle NaN rows
        if keep_nanapis:
            df = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_with_nanapis.csv'))
        else:
            df = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_without_nanapis.csv'))

        non_hydrocarbons = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']
        df['non_hydrocarbon_sum'] = df[non_hydrocarbons].sum(axis=1, skipna=True)

        # Filter rows based on threshold
        filtered_df = df[df['non_hydrocarbon_sum'] < threshold]

        # Clean up before returning
        return filtered_df.drop(columns=['non_hydrocarbon_sum'])

    # Return a callable structure
    return filter_dataframe

filter_func = create_data_structure()

raw_data_usgs = filter_func(1000, True).copy()

components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']


year_usgs_dict_all = {
    comp: {k: v for k, v in raw_data_usgs[raw_data_usgs[comp].notna()]
           .groupby('BASIN_NAME')['Year'].max().to_dict().items()}
    for comp in components
}


components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
result_usgs_filtered = filter_to_closest_years_by_xy_and_component(result_usgs, components, year_usgs_dict_all)



def propagate_uncertainties_normalization(data_on_grid):
    """
    Normalize non-C1/CO2 components to 100 - C1 - CO2 and propagate 1σ uncertainties.
    Negative values are avoided. If C1 + CO2 > 100, other components are set to 0.
    Uses simplified uncertainty propagation to avoid unrealistic uncertainty inflation.
    """
    comps_all = ['HE','CO2','H2','N2','H2S','AR','O2',
                 'C1','C2','C3','N-C4','I-C4','N-C5','I-C5','C6+']
    comps_fixed = ['C1', 'CO2']
    comps_renorm = [c for c in comps_all if c not in comps_fixed]
    
    # values
    A = data_on_grid[comps_renorm].to_numpy(dtype=float)
    σ = data_on_grid[['std_' + c for c in comps_renorm]].to_numpy(dtype=float)

    A_total = np.nansum(A, axis=1)
    C1 = data_on_grid['C1'].to_numpy(dtype=float)
    CO2 = data_on_grid['CO2'].to_numpy(dtype=float)
    se_C1 = data_on_grid['std_C1'].to_numpy(dtype=float)
    se_CO2 = data_on_grid['std_CO2'].to_numpy(dtype=float)

    # Cap C1 + CO2 at 100 and compute remaining fraction
    sum_C1_CO2 = C1 + CO2
    sum_C1_CO2_clipped = np.minimum(sum_C1_CO2, 100)
    rem = 100 - sum_C1_CO2_clipped

    # renormalize if rem > 0, else set everything to 0
    F = np.zeros_like(A)
    abs_err = np.zeros_like(σ)

    valid = (rem > 0) & (A_total > 0)
    if np.any(valid):
        A_valid = A[valid]
        rem_valid = rem[valid]
        A_total_valid = A_total[valid]
        σ_valid = σ[valid]

        # Renormalized values
        scale = rem_valid[:, None] / A_total_valid[:, None]
        F_valid = A_valid * scale
        abs_err_valid = σ_valid * scale

        # Clip to avoid negative values due to numerical noise
        F_valid = np.clip(F_valid, 0, None)

        F[valid] = F_valid
        abs_err[valid] = abs_err_valid

    # write results back
    for i, c in enumerate(comps_renorm):
        data_on_grid[c] = F[:, i]
        data_on_grid['std_' + c] = abs_err[:, i]

    # C1 and CO2 unchanged, but clipped so sum ≤ 100
    C1_clipped = np.clip(C1, 0, 100)
    CO2_clipped = np.clip(CO2, 0, 100 - C1_clipped)

    data_on_grid['C1'] = C1_clipped
    data_on_grid['std_C1'] = se_C1
    data_on_grid['CO2'] = CO2_clipped
    data_on_grid['std_CO2'] = se_CO2

    # sum check
    data_on_grid['sum_check'] = data_on_grid[comps_all].sum(axis=1).round(5)

    return data_on_grid


ghgrp_subset = result_ghgrp_filtered[['X', 'Y', 'BASIN_NAME', 'C1', 'std_C1', 'CO2', 'std_CO2']].copy()

combined_df = result_usgs_filtered.copy().drop(columns=['C1', 'std_C1', 'CO2', 'std_CO2'], errors='ignore')  # avoid duplication if already present

combined_df = combined_df.merge(ghgrp_subset, on=['X', 'Y', 'BASIN_NAME'], how='left')

components = ['C1', 'CO2', 'HE', 'H2', 'N2', 'H2S', 'AR', 'O2',
              'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
basins = combined_df['BASIN_NAME'].dropna().unique()
output_dir = "datasets_leaflet_new"

for basin in basins:
    basin_sanitized = basin.replace(" ", "_").replace("/", "_")
    basin_dir = os.path.join(output_dir, basin_sanitized)
    os.makedirs(basin_dir, exist_ok=True)

    df_basin = combined_df[combined_df['BASIN_NAME'] == basin].copy()

    for comp in components:
        for suffix in ['', 'std_']:
            col = suffix + comp
            if col not in df_basin.columns:
                continue
            df_sub = df_basin[['X', 'Y', col]].dropna().copy()
            df_sub = df_sub.rename(columns={col: 'value'})
            out_path = os.path.join(basin_dir, f"dataset_{col}.csv")
            df_sub.to_csv(out_path, index=False)


