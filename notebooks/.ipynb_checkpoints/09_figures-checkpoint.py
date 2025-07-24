import pandas as pd
import numpy as np
import os

def process_ghgrp_processing_data(root_path):
    """
    Compute weighted average and bootstrap standard errors for CH4 and CO2 mole fractions
    from GHGRP processing plant data, grouped by basin and year, and by basin overall.
    """
    # Load data
    ghgrp_processing_gdf = pd.read_csv(os.path.join(root_path, 'ghgrp_processing_new_weighted_variable_gdf.csv'))

    def weighted_average(group, value_column, weight_column):
        weighted_sum = (group[value_column] * group[weight_column]).sum()
        weight_sum = group[weight_column].sum()
        return weighted_sum / weight_sum if weight_sum != 0 else np.nan

    def bootstrap_se(df, value_column, weight_column, B=1000):
        bootstrap_means = []
        n = len(df)
        for _ in range(B):
            sample = df.sample(n, replace=True)
            weighted_mean = weighted_average(sample, value_column, weight_column)
            bootstrap_means.append(weighted_mean)
        bootstrap_means = np.array(bootstrap_means)
        bootstrap_mean = np.mean(bootstrap_means)
        se_bootstrap = np.sqrt(np.sum((bootstrap_means - bootstrap_mean) ** 2) / (B - 1))
        return se_bootstrap

    # Basin-Year Aggregation
    weighted_results = ghgrp_processing_gdf.groupby(['BASIN_NAME', 'Year']).apply(
        lambda group: pd.Series({
            'C1': weighted_average(group, 'ch4_average_mole_fraction', 'Plant_Flow'),
            'CO2': weighted_average(group, 'co2_average_mole_fraction', 'Plant_Flow'),
        })
    ).reset_index()

    bootstrap_results = []
    for _, row in weighted_results.iterrows():
        basin_name = row['BASIN_NAME']
        year = row['Year']
        subset = ghgrp_processing_gdf[(ghgrp_processing_gdf['BASIN_NAME'] == basin_name) & (ghgrp_processing_gdf['Year'] == year)]
        se_ch4 = bootstrap_se(subset, 'ch4_average_mole_fraction', 'Plant_Flow')
        se_co2 = bootstrap_se(subset, 'co2_average_mole_fraction', 'Plant_Flow')
        bootstrap_results.append({
            'BASIN_NAME': basin_name,
            'Year': year,
            'se_C1': se_ch4,
            'se_CO2': se_co2
        })
    bootstrap_df = pd.DataFrame(bootstrap_results)
    results_ghgrp_proc_by_basin_year = pd.merge(weighted_results, bootstrap_df, on=['BASIN_NAME', 'Year'])

    # Basin-only Aggregation
    weighted_results_basin = ghgrp_processing_gdf.groupby('BASIN_NAME').apply(
        lambda group: pd.Series({
            'C1': weighted_average(group, 'ch4_average_mole_fraction', 'Plant_Flow'),
            'CO2': weighted_average(group, 'co2_average_mole_fraction', 'Plant_Flow'),
            'Gas': group['Plant_Flow'].sum(),
            'Year': 2017
        })
    ).reset_index()

    bootstrap_results_basin = []
    for _, row in weighted_results_basin.iterrows():
        basin_name = row['BASIN_NAME']
        subset = ghgrp_processing_gdf[ghgrp_processing_gdf['BASIN_NAME'] == basin_name]
        se_ch4 = bootstrap_se(subset, 'ch4_average_mole_fraction', 'Plant_Flow')
        se_co2 = bootstrap_se(subset, 'co2_average_mole_fraction', 'Plant_Flow')
        bootstrap_results_basin.append({
            'BASIN_NAME': basin_name,
            'se_C1': se_ch4,
            'se_CO2': se_co2
        })
    bootstrap_df_basin = pd.DataFrame(bootstrap_results_basin)
    results_ghgrp_proc_by_basin = pd.merge(weighted_results_basin, bootstrap_df_basin, on='BASIN_NAME')
    # process for future use

    ghgrp_processing_gdf = ghgrp_processing_gdf.rename(columns={'Plant_Flow':'Gas', 'ch4_average_mole_fraction':'C1', 'co2_average_mole_fraction':'CO2'})
    ghgrp_processing_gdf['C1'] /= 100
    ghgrp_processing_gdf['CO2'] /= 100

    ghgrp_processing_gdf = ghgrp_processing_gdf[['Gas',
       'C1', 'CO2', 'BASIN_NAME', 'latitude', 'longitude', 'Count', 'Year']]
    # Add shales
    
    shales_gdf = gpd.read_file(shapefiles_path + '/shale_plays')
    haynesville_shale_gdf = shales_gdf[shales_gdf.Shale_play == 'Haynesville-Bossier']
    ghgrp_processing_gdf = gpd.GeoDataFrame(
        ghgrp_processing_gdf, 
        geometry=gpd.points_from_xy(ghgrp_processing_gdf.longitude, ghgrp_processing_gdf.latitude)
    )
    ghgrp_processing_gdf = ghgrp_processing_gdf.set_crs("EPSG:4326", allow_override=True)  # assuming original CRS is EPSG:4326
    
    ghgrp_processing_gdf = ghgrp_processing_gdf.to_crs(epsg=26914)
    
    # Convert shales_gdf from EPSG:3857 (Web Mercator) to EPSG:26914 (UTM Zone 14N)
    haynesville_shale_gdf = haynesville_shale_gdf.to_crs(epsg=26914)
    
    # Step 4: Perform the spatial join (check if points are within polygons)
    ghgrp_processing_joined_shales_gdf = gpd.sjoin(ghgrp_processing_gdf, haynesville_shale_gdf, how="inner", predicate="within")

    
    # Step 1: Group by BASIN_NAME and Year and compute weighted averages
    weighted_results = ghgrp_processing_joined_shales_gdf.groupby(['Shale_play', 'Year']).apply(
        lambda group: pd.Series({
            'C1': weighted_average(group, 'C1', 'Gas'),
            'CO2': weighted_average(group, 'CO2', 'Gas'),
        })
    ).reset_index()
    
    # Step 2: Compute bootstrap standard error for each group
    bootstrap_results = []
    for _, row in weighted_results.iterrows():
        basin_name = row['Shale_play']
        year = row['Year']
        
        # Filter the original dataframe for the current basin and year
        subset = ghgrp_processing_joined_shales_gdf[(ghgrp_processing_joined_shales_gdf['Shale_play'] == basin_name) & (ghgrp_processing_joined_shales_gdf['Year'] == year)]
        
        # Bootstrap standard error for CH4
        se_ch4 = bootstrap_se(subset, 'C1', 'Gas')
        # Bootstrap standard error for CO2
        se_co2 = bootstrap_se(subset, 'CO2', 'Gas')
        
        bootstrap_results.append({
            'Shale_play': basin_name,
            'Year': year,
            'se_C1': se_ch4,
            'se_CO2': se_co2
        })
    
    # Convert bootstrap results to a DataFrame
    bootstrap_df = pd.DataFrame(bootstrap_results)
    
    # Merge weighted results with bootstrap uncertainties
    results_ghgrp_proc_by_shale_year = pd.merge(weighted_results, bootstrap_df, on=['Shale_play', 'Year'])
    
    # Step 1: Group by BASIN_NAME and compute weighted averages
    
    weighted_results_shale = ghgrp_processing_joined_shales_gdf.groupby('Shale_play').apply(
        lambda group: pd.Series({
            'C1': weighted_average(group, 'C1', 'Gas'),
            'CO2': weighted_average(group, 'CO2', 'Gas'),
            'Gas': group['Gas'].sum(),  # total flow
            'Year': 2017  # constant value
        })
    ).reset_index()
    
    # Step 2: Compute bootstrap standard error for each basin
    bootstrap_results_shale = []
    for _, row in weighted_results_shale.iterrows():
        basin_name = row['Shale_play']
        
        # Filter the original dataframe for the current basin
        subset = ghgrp_processing_joined_shales_gdf[ghgrp_processing_joined_shales_gdf['Shale_play'] == basin_name]
        
        # Bootstrap standard error for CH4
        se_ch4 = bootstrap_se(subset, 'C1', 'Gas')
        # Bootstrap standard error for CO2
        se_co2 = bootstrap_se(subset, 'CO2', 'Gas')
        
        bootstrap_results_shale.append({
            'Shale_play': basin_name,
            'se_C1': se_ch4,
            'se_CO2': se_co2
        })
    
    # Convert bootstrap results to a DataFrame
    bootstrap_df_shale = pd.DataFrame(bootstrap_results_shale)
    
    # Merge weighted results with bootstrap uncertainties
    results_ghgrp_proc_by_shale = pd.merge(weighted_results_shale, bootstrap_df_shale, on='Shale_play')

    results_ghgrp_proc_by_shale['C1'] *= 100
    results_ghgrp_proc_by_shale['CO2'] *= 100
    return results_ghgrp_proc_by_basin, ghgrp_processing_gdf, results_ghgrp_proc_by_shale

results_ghgrp_proc_by_basin, ghgrp_processing_gdf, results_ghgrp_proc_by_shale = process_ghgrp_processing_data(root_path)

def propagate_uncertainties_normalization(data_on_grid):
    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']

    # Generate list of standard deviation columns for each component
    list_stds = ['std_' + comp for comp in components]

    # Extract values and their uncertainties
    values = data_on_grid[components].values
    uncertainties = data_on_grid[list_stds].values

    # Step 1: Calculate the Sum and its Uncertainty
    sum_values = np.nansum(values, axis=1)
    uncertainty_sum = np.sqrt(np.nansum(uncertainties**2, axis=1))

    # Step 2: Normalize the Values to sum to 100 and add to DataFrame
    for i, comp in enumerate(components):
        data_on_grid[comp] = values[:, i] / sum_values

    # Step 3: Calculate the Fractional Uncertainty of Each Normalized Value
    for i, comp in enumerate(components):
        fractional_uncertainty = np.sqrt((uncertainties[:, i] / values[:, i])**2 + (uncertainty_sum / sum_values)**2)
        data_on_grid['std_' + comp] = fractional_uncertainty

    # Step 4: Calculate the Final Absolute Uncertainty for normalized values and add to DataFrame
    for i, comp in enumerate(components):
        absolute_uncertainty =  data_on_grid['std_' + comp] * data_on_grid[comp] / sum_values
        data_on_grid['std_' + comp] = absolute_uncertainty

    # Step 5: Add a column to check if the sum of the normalized values is 100
    data_on_grid['sum_check'] = data_on_grid[components].sum(axis=1).round(2)

    return data_on_grid


def propagate_uncertainties_normalization_mean(data_on_grid):
    components = ['C1', 'C2', 'HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
    mean_components = [f"{comp}_mean" for comp in components]


    # Generate list of standard deviation columns for each component
    list_stds = ['std_' + comp for comp in mean_components]

    # Extract values and their uncertainties
    values = data_on_grid[mean_components].values
    uncertainties = data_on_grid[list_stds].values

    # Step 1: Calculate the Sum and its Uncertainty
    sum_values = np.nansum(values, axis=1)
    uncertainty_sum = np.sqrt(np.nansum(uncertainties**2, axis=1))

    # Step 2: Normalize the Values to sum to 100 and add to DataFrame
    for i, comp in enumerate(mean_components):
        data_on_grid[comp] = values[:, i] / sum_values

    # Step 3: Calculate the Fractional Uncertainty of Each Normalized Value
    for i, comp in enumerate(mean_components):
        fractional_uncertainty = np.sqrt((uncertainties[:, i] / values[:, i])**2 + (uncertainty_sum / sum_values)**2)
        data_on_grid['std_' + comp] = fractional_uncertainty

    # Step 4: Calculate the Final Absolute Uncertainty for normalized values and add to DataFrame
    for i, comp in enumerate(mean_components):
        absolute_uncertainty =  data_on_grid['std_' + comp] * data_on_grid[comp] / sum_values
        data_on_grid['std_' + comp] = absolute_uncertainty

    # Step 5: Add a column to check if the sum of the normalized values is 100
#     data_on_grid['sum_check_mean'] = data_on_grid[components].sum(axis=1).round(2)

    return data_on_grid


def predic_beta(pred1, pred_1_std, pred2, beta):
    weight_krig = 1
    weight_pred = beta * (pred_1_std ** 2)
    sum_weights = weight_krig + weight_pred
    prediction = (pred1 + weight_pred * pred2) / sum_weights
    return prediction

def filter_non_hydrocarbons(df, threshold):
    non_hydrocarbons = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']
    df['non_hydrocarbon_sum'] = df[non_hydrocarbons].fillna(0).sum(axis=1)
    filtered_df = df[df['non_hydrocarbon_sum'] < threshold]
    filtered_df = filtered_df.drop(columns=['non_hydrocarbon_sum'])
    return filtered_df

def calculate_weighted_mean(group, value_column, weight_column):
    weighted_sum = (group[value_column] * group[weight_column]).sum()
    weight_sum = group[weight_column].sum()
    return weighted_sum / weight_sum if weight_sum != 0 else np.nan


def compute_weighted_mean_by_year(df_list, value_column, weight_column, basin_column, year_column):
    basin_year_data_dict = {}
    
    for df in df_list:
        # Compute weighted mean
        weighted_means = (
            df.groupby([basin_column, year_column])
            .apply(lambda group: calculate_weighted_mean(group, value_column, weight_column))
            .reset_index(name='weighted_mean')
        )
        
        # Compute total Oil and Gas
        oil_gas_sums = (
            df.groupby([basin_column, year_column])[['Oil', 'Gas']].sum().reset_index()
        )

        # Merge both results
        merged = weighted_means.merge(oil_gas_sums, on=[basin_column, year_column])

        # Populate final dictionary
        for _, row in merged.iterrows():
            key = (row[basin_column], row[year_column])
            if key not in basin_year_data_dict:
                basin_year_data_dict[key] = []
            basin_year_data_dict[key].append({
                'weighted_mean': row['weighted_mean'],
                'Oil': row['Oil'],
                'Gas': row['Gas']
            })

    return basin_year_data_dict

def compute_combined_results(basin_year_means, component_name):
    results = []
    
    for basin in set(b for b, _ in basin_year_means.keys()):
        year_data = {year: basin_year_means[(basin, year)] 
                     for b, year in basin_year_means.keys() if b == basin}
        
#         closest_years = sorted([y for y in year_data.keys() if y >= 2010], reverse=True)
        
        closest_years = sorted(year_data.keys())

        for year in closest_years:
            entries = year_data[year]  # List of dicts: each has weighted_mean, Oil, Gas
            
            if entries:
                means = [e['weighted_mean'] for e in entries]
                oils = [e['Oil'] for e in entries]
                gases = [e['Gas'] for e in entries]

                mean_value = np.mean(means)
                std_error = np.std(means, ddof=1) / np.sqrt(len(means))
                total_oil = np.mean(oils)
                total_gas = np.mean(gases)
            else:
                mean_value = np.nan
                std_error = np.nan
                total_oil = np.nan
                total_gas = np.nan

            results.append({
                'BASIN_NAME': basin,
                'Year': year,
                f'{component_name}': mean_value,
                f'se_{component_name}': std_error,
                'Oil': total_oil,
                'Gas': total_gas
            })
    
    return pd.DataFrame(results)


def upload_result_simple(path):
    csv_files_pattern = os.path.join(root_path, 'out', path)
    csv_files = glob.glob(csv_files_pattern)
    result_all_list = [pd.read_csv(file) for file in csv_files]
    result_all = pd.concat(result_all_list)

    result_all['X'] = (result_all['X'] * 6.34).round(1)
    result_all['Y'] = (result_all['Y'] * 6.34).round(1)
    return result_all

def return_results_df_usgs_combined(result_dic, a, b, threshold, keep_nanapis, radius_factor, propagate, C2_predicted=False):
    all_prod_indices_all = pd.read_csv(f'all_prod_indices_all_{threshold}.csv')
    all_prod_indices_all['X'] = (all_prod_indices_all['X']).round(1)
    all_prod_indices_all['Y'] = (all_prod_indices_all['Y']).round(1)

    all_prod_indices_all['Prod_X_grid'] = all_prod_indices_all['Prod_X_grid'].round(1)
    all_prod_indices_all['Prod_Y_grid'] = all_prod_indices_all['Prod_Y_grid'].round(1)
    
    print('Created all_prod_indices_all...')
    data_source = 'usgs'
    sample = 2
    radius_factor = 0.175
    result_usgs = result_dic[(threshold, a, b)]
    result_usgs['X'] = (result_usgs['X']).round(1)
    result_usgs['Y'] = (result_usgs['Y']).round(1)
    print('Loaded results...')
    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']

    rename_mean = {comp: f"{comp}_mean" for comp in components}
    rename_std = {f"std_{comp}": f"std_{comp}_mean" for comp in components}

    result_usgs = result_usgs.rename(columns={**rename_mean, **rename_std})


    list_result_all_sgs = []
    for i in range(1, 6):
        a = 2
        b = 2
        interp_method = 'sgs_' + str(i)
        'Uploading sgs...'
        result_all_sgs = upload_result_simple(
                                   f'result_data_source_{data_source}_a_{a}_b_{b}_basin_*_sample_{sample}_interp_{interp_method}_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')
        if C2_predicted:
            result_all_sgs = result_all_sgs.drop(columns=['std_C1', 'std_C2'])

            print('Merging sgs...')
            
            base_cols = ['X', 'Y', 'T', 'C1_predic_gor', 'C2_predic_gor', 'beta_C1', 'beta_C2']
            std_cols = [f"std_{comp}_mean" for comp in components]
            mean_cols = [f"{comp}_mean" for comp in components]
                        
            subset = result_usgs[base_cols + mean_cols + std_cols]

            result_all_sgs = pd.merge(result_all_sgs,
                                      subset,
                                     on=['X','Y','T'])
            result_all_sgs = result_all_sgs[~pd.isna(result_all_sgs['C1_predic_gor'])]
            print('Adding predictions...')
            result_all_sgs['std_C1'] = result_all_sgs['std_C1_mean']
            result_all_sgs['std_C2'] = result_all_sgs['std_C2_mean']

            result_all_sgs['C1'] = predic_beta(result_all_sgs['C1'], result_all_sgs['std_C1'], result_all_sgs['C1_predic_gor'], result_all_sgs['beta_C1'].iloc[0])
            
#             result_all_sgs['C1_mean'] = predic_beta(result_all_sgs['C1_mean'], result_all_sgs['std_C1'], result_all_sgs['C1_predic_gor'], result_all_sgs['beta_C1'].iloc[0])

            result_all_sgs['C2'] = predic_beta(result_all_sgs['C2'], result_all_sgs['std_C2'], result_all_sgs['C2_predic_gor'], result_all_sgs['beta_C2'].iloc[0])
#             result_all_sgs['C2_mean'] = predic_beta(result_all_sgs['C2_mean'], result_all_sgs['std_C2'], result_all_sgs['C2_predic_gor'], result_all_sgs['beta_C2'].iloc[0])

        list_result_all_sgs.append(result_all_sgs)

    new_list_result_all_sgs = []
    for result_all_sgs in list_result_all_sgs:
        print('Propagating uncertainties...')
        if propagate:

            result_all_sgs = propagate_uncertainties_normalization_mean(result_all_sgs)
            result_all_sgs = propagate_uncertainties_normalization(result_all_sgs)
        reference_date = '1918-08-19T00:00:00.000000000'
        result_all_sgs['Year'] = (pd.Timestamp(reference_date) + pd.to_timedelta(result_all_sgs['T'], unit='D')).dt.year


        new_list_result_all_sgs.append(result_all_sgs)

    all_results = []
    mean_cols = [f"{comp}_mean" for comp in components]
    for component in components + mean_cols:
  
        # compute weighted mean by year and component
        basin_year_means = compute_weighted_mean_by_year(
            new_list_result_all_sgs, value_column=component, weight_column='Gas', 
            basin_column='BASIN_NAME', year_column='Year'
        )
        component_results = compute_combined_results(basin_year_means, component)
        all_results.append(component_results)

    # Merge results for all components into a single DataFrame
    results_df_usgs_combined = all_results[0]
    for component_results in all_results[1:]:
        results_df_usgs_combined = results_df_usgs_combined.merge(
            component_results, 
            on=['BASIN_NAME', 'Year'], 
            how='outer', 
            suffixes=('', '_dup')
        )
        # Remove duplicate suffix columns for Gas and Oil
        if 'Gas_dup' in results_df_usgs_combined.columns:
            results_df_usgs_combined['Gas'] = results_df_usgs_combined[['Gas', 'Gas_dup']].sum(axis=1, skipna=True)
            results_df_usgs_combined.drop(columns=['Gas_dup'], inplace=True)
        if 'Oil_dup' in results_df_usgs_combined.columns:
            results_df_usgs_combined['Oil'] = results_df_usgs_combined[['Oil', 'Oil_dup']].sum(axis=1, skipna=True)
            results_df_usgs_combined.drop(columns=['Oil_dup'], inplace=True)

    return results_df_usgs_combined

# save average results by year by basin for figures for usgs
radius_factor = 0.175
C2_predicted = True
data_source = "usgs" 
filename = f"result_dic_r0175_{data_source}.pkl"

with open(filename, "rb") as f:
    result_dic = pickle.load(f)
    

# Initialize the dictionary with tqdm for progress monitoring
results_df_usgs_combined_dict = {}
for (a, b) in tqdm(zip([-1, 1, -1, 1], [-1, 1, 1, -1]), desc="Pairs of (a, b)", total=4, position=0):
    for threshold in tqdm([1000, 10, 25, 50], desc="Threshold", position=1, leave=False):
        for keep_nanapis in tqdm([True], desc="Keep NaNs", position=2, leave=False):
            for propagate in tqdm([True], desc="Propagate", position=3, leave=False):
                results_df_usgs_combined_dict[(a, b, threshold, keep_nanapis, propagate)] = return_results_df_usgs_combined(
                   result_dic, a, b, threshold, keep_nanapis, radius_factor, propagate, C2_predicted
                )
with open("results_df_usgs_combined_dict_C2_predicted.pkl", "wb") as f:
    pickle.dump(results_df_usgs_combined_dict, f)


def return_results_df_ghgrp_combined(result_dic, a, b, threshold, keep_nanapis, radius_factor):
    all_prod_indices_all = pd.read_csv(f'all_prod_indices_all_ghgrp.csv')
    all_prod_indices_all['X'] = (all_prod_indices_all['X']).round(1)
    all_prod_indices_all['Y'] = (all_prod_indices_all['Y']).round(1)

    all_prod_indices_all['Prod_X_grid'] = all_prod_indices_all['Prod_X_grid'].round(1)
    all_prod_indices_all['Prod_Y_grid'] = all_prod_indices_all['Prod_Y_grid'].round(1)
    
    print('Created all_prod_indices_all...')
    data_source = 'ghgrp'
    sample = 2
    radius_factor = 0.15
    result_ghgrp = result_dic[(threshold, a, b)]
    result_ghgrp['X'] = (result_ghgrp['X']).round(1)
    result_ghgrp['Y'] = (result_ghgrp['Y']).round(1)
    print('Loaded results...')
    components = ['CO2', 'C1']

    rename_mean = {comp: f"{comp}_mean" for comp in components}
    rename_std = {f"std_{comp}": f"std_{comp}_mean" for comp in components}

    result_ghgrp = result_ghgrp.rename(columns={**rename_mean, **rename_std})    
    
    
    list_result_all_sgs = []
    for i in range(1, 4):
        a = 2
        b = 2
        interp_method = 'sgs_' + str(i)
        'Uploading sgs...'
        result_all_sgs = upload_result_simple(
                                   f'result_data_source_{data_source}_a_{a}_b_{b}_basin_*_sample_{sample}_interp_{interp_method}_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')

        result_all_sgs = result_all_sgs.drop(columns=['std_C1'])
        
        
        base_cols = ['X', 'Y', 'T', 'C1_predic_gor', 'beta_C1']
        std_cols = [f"std_{comp}_mean" for comp in components]
        mean_cols = [f"{comp}_mean" for comp in components]

        subset = result_ghgrp[base_cols + mean_cols + std_cols]

        result_all_sgs = pd.merge(result_all_sgs,
                                  subset,
                                 on=['X','Y','T'])
        result_all_sgs = result_all_sgs[~pd.isna(result_all_sgs['C1_predic_gor'])]
        print('Adding predictions...')
        result_all_sgs['std_C1'] = result_all_sgs['std_C1_mean']
        
        result_all_sgs = result_all_sgs[~pd.isna(result_all_sgs['C1_predic_gor'])]
        result_all_sgs['C1'] = predic_beta(result_all_sgs['C1'], result_all_sgs['std_C1'], result_all_sgs['C1_predic_gor'], result_all_sgs['beta_C1'].iloc[0])
#         result_all_sgs['C1_mean'] = predic_beta(result_all_sgs['C1_mean'], result_all_sgs['std_C1'], result_all_sgs['C1_predic_gor'], result_all_sgs['beta_C1'].iloc[0])

        reference_date = '1918-08-19T00:00:00.000000000'
        result_all_sgs['Year'] = (pd.Timestamp(reference_date) + pd.to_timedelta(result_all_sgs['T'], unit='D')).dt.year

        list_result_all_sgs.append(result_all_sgs)

    new_list_result_all_sgs = list_result_all_sgs

    all_results = []
    mean_cols = [f"{comp}_mean" for comp in components]

    for component in components + mean_cols:

        # compute weighted mean by year and component
        basin_year_means = compute_weighted_mean_by_year(
            new_list_result_all_sgs, value_column=component, weight_column='Gas', 
            basin_column='BASIN_NAME', year_column='Year'
        )
        # compute overall mean across sgs --> CHANGE THAT BECAUSE FOR NOW ONLY MEAN, NOT WEIGHTED MEAN.
        component_results = compute_combined_results(basin_year_means, component)
        all_results.append(component_results)

    # Merge results for all components into a single DataFrame
    results_df_ghgrp_combined = all_results[0]
    for component_results in all_results[1:]:
        results_df_ghgrp_combined = results_df_ghgrp_combined.merge(
            component_results, 
            on=['BASIN_NAME', 'Year'], 
            how='outer', 
            suffixes=('', '_dup')
        )
        # Remove duplicate suffix columns for Gas and Oil
        if 'Gas_dup' in results_df_ghgrp_combined.columns:
            results_df_ghgrp_combined['Gas'] = results_df_ghgrp_combined[['Gas', 'Gas_dup']].sum(axis=1, skipna=True)
            results_df_ghgrp_combined.drop(columns=['Gas_dup'], inplace=True)
        if 'Oil_dup' in results_df_ghgrp_combined.columns:
            results_df_ghgrp_combined['Oil'] = results_df_ghgrp_combined[['Oil', 'Oil_dup']].sum(axis=1, skipna=True)
            results_df_ghgrp_combined.drop(columns=['Oil_dup'], inplace=True)

    return results_df_ghgrp_combined

# save average results by year by basin for figures for ghgrp
data_source = "ghgrp" 
filename = f"result_dic_r0175_{data_source}.pkl"

with open(filename, "rb") as f:
    result_dic = pickle.load(f)
    
radius_factor = 0.15
# Initialize the dictionary with tqdm for progress monitoring
results_df_ghgrp_combined_dict = {}
for (a, b) in tqdm(zip([-1, 1, -1, 1], [-1, 1, 1, -1]), desc="Pairs of (a, b)", total=4, position=0):
    for threshold in tqdm([1000], desc="Threshold", position=1, leave=False):
        for keep_nanapis in tqdm([True], desc="Keep NaNs", position=2, leave=False):
            print((a,b,threshold))
            results_df_ghgrp_combined_dict[(a, b, threshold, keep_nanapis)] = return_results_df_ghgrp_combined(
               result_dic, a, b, threshold, keep_nanapis, radius_factor
            )
with open("results_df_ghgrp_combined_dict.pkl", "wb") as f:
    pickle.dump(results_df_ghgrp_combined_dict, f)




def compute_combined_results_shales(basin_year_means, component_name):
    results = []
    
    for basin in set(b for b, _ in basin_year_means.keys()):
        year_data = {year: basin_year_means[(basin, year)] 
                     for b, year in basin_year_means.keys() if b == basin}
        
#         closest_years = sorted([y for y in year_data.keys() if y >= 2010], reverse=True)
        
        closest_years = sorted(year_data.keys())

        for year in closest_years:
            entries = year_data[year]  # List of dicts: each has weighted_mean, Oil, Gas
            
            if entries:
                means = [e['weighted_mean'] for e in entries]
                oils = [e['Oil'] for e in entries]
                gases = [e['Gas'] for e in entries]

                mean_value = np.mean(means)
                std_error = np.std(means, ddof=1) / np.sqrt(len(means))
                total_oil = np.mean(oils)
                total_gas = np.mean(gases)
            else:
                mean_value = np.nan
                std_error = np.nan
                total_oil = np.nan
                total_gas = np.nan

            results.append({
                'Shale_play': basin,
                'Year': year,
                f'{component_name}': mean_value,
                f'se_{component_name}': std_error,
                'Oil': total_oil,
                'Gas': total_gas
            })
    
    return pd.DataFrame(results)



def return_results_df_usgs_combined_shales(result_dic, a, b, threshold, keep_nanapis, radius_factor, propagate, C2_predicted=False):
    all_prod_indices_all = pd.read_csv(f'all_prod_indices_all_{threshold}.csv')
    all_prod_indices_all['X'] = (all_prod_indices_all['X']).round(1)
    all_prod_indices_all['Y'] = (all_prod_indices_all['Y']).round(1)

    all_prod_indices_all['Prod_X_grid'] = all_prod_indices_all['Prod_X_grid'].round(1)
    all_prod_indices_all['Prod_Y_grid'] = all_prod_indices_all['Prod_Y_grid'].round(1)
    
    print('Created all_prod_indices_all...')
    data_source = 'usgs'
    sample = 2
    radius_factor = 0.175
    result_usgs = result_dic[(threshold, a, b)]
    result_usgs['X'] = (result_usgs['X']).round(1)
    result_usgs['Y'] = (result_usgs['Y']).round(1)
    
    result_usgs = gpd.GeoDataFrame(
    result_usgs, 
    geometry=gpd.points_from_xy(result_usgs.X, result_usgs.Y)
    )
    result_usgs = gpd.sjoin(result_usgs, haynesville_shale_gdf, how="inner", predicate="within")
        
    print('Loaded results...')
    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']

    rename_mean = {comp: f"{comp}_mean" for comp in components}
    rename_std = {f"std_{comp}": f"std_{comp}_mean" for comp in components}

    result_usgs = result_usgs.rename(columns={**rename_mean, **rename_std})


    list_result_all_sgs = []
    for i in range(1, 4):
        a = 2
        b = 2
        interp_method = 'sgs_' + str(i)
        'Uploading sgs...'
        result_all_sgs = upload_result_simple(
                                   f'result_data_source_{data_source}_a_{a}_b_{b}_basin_*_sample_{sample}_interp_{interp_method}_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')
        
        result_all_sgs = gpd.GeoDataFrame(
        result_all_sgs, 
        geometry=gpd.points_from_xy(result_all_sgs.X, result_all_sgs.Y)
        )
        result_all_sgs = gpd.sjoin(result_all_sgs, haynesville_shale_gdf, how="inner", predicate="within")
        print(len(result_all_sgs))
        result_all_sgs = result_all_sgs.drop(columns=['std_C1', 'std_C2'])

        print('Merging sgs...')

        base_cols = ['X', 'Y', 'T', 'C1_predic_gor', 'C2_predic_gor', 'beta_C1', 'beta_C2']
        std_cols = [f"std_{comp}_mean" for comp in components]
        mean_cols = [f"{comp}_mean" for comp in components]

        subset = result_usgs[base_cols + mean_cols + std_cols]

        result_all_sgs = pd.merge(result_all_sgs,
                                  subset,
                                 on=['X','Y','T'])
        result_all_sgs = result_all_sgs[~pd.isna(result_all_sgs['C1_predic_gor'])]
        print('Adding predictions...')
        result_all_sgs['std_C1'] = result_all_sgs['std_C1_mean']
        result_all_sgs['std_C2'] = result_all_sgs['std_C2_mean']

        result_all_sgs['C1'] = predic_beta(result_all_sgs['C1'], result_all_sgs['std_C1'], result_all_sgs['C1_predic_gor'], result_all_sgs['beta_C1'].iloc[0])

#         result_all_sgs['C1_mean'] = predic_beta(result_all_sgs['C1_mean'], result_all_sgs['std_C1'], result_all_sgs['C1_predic_gor'], result_all_sgs['beta_C1'].iloc[0])

        result_all_sgs['C2'] = predic_beta(result_all_sgs['C2'], result_all_sgs['std_C2'], result_all_sgs['C2_predic_gor'], result_all_sgs['beta_C2'].iloc[0])
#         result_all_sgs['C2_mean'] = predic_beta(result_all_sgs['C2_mean'], result_all_sgs['std_C2'], result_all_sgs['C2_predic_gor'], result_all_sgs['beta_C2'].iloc[0])

        list_result_all_sgs.append(result_all_sgs)

    new_list_result_all_sgs = []
    for result_all_sgs in list_result_all_sgs:
        print('Propagating uncertainties...')
        if propagate:
            result_all_sgs = propagate_uncertainties_normalization_mean(result_all_sgs)
            result_all_sgs = propagate_uncertainties_normalization(result_all_sgs)
        reference_date = '1918-08-19T00:00:00.000000000'
        result_all_sgs['Year'] = (pd.Timestamp(reference_date) + pd.to_timedelta(result_all_sgs['T'], unit='D')).dt.year
        def keep_closest_year(df, year_col='Year', group_cols=['X', 'Y'], target_year=2017):
            return df.loc[df.groupby(group_cols)[year_col].apply(lambda x: (x - target_year).abs().idxmin()).values]

        result_all_sgs = keep_closest_year(result_all_sgs.reset_index(drop=True), year_col='Year', group_cols=['X', 'Y'], target_year=2017)

        new_list_result_all_sgs.append(result_all_sgs)

    all_results = []
    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']

    mean_cols = [f"{comp}_mean" for comp in components]

    for component in components + mean_cols:
        # compute weighted mean by year and component
        basin_year_means = compute_weighted_mean_by_year(
            new_list_result_all_sgs, value_column=component, weight_column='Gas', 
            basin_column='Shale_play', year_column='Year'
        )
        # compute overall mean across sgs --> CHANGE THAT BECAUSE FOR NOW ONLY MEAN, NOT WEIGHTED MEAN.
        component_results = compute_combined_results_shales(basin_year_means, component)
        all_results.append(component_results)

    # Merge results for all components into a single DataFrame
    results_df_usgs_combined = all_results[0]
    for component_results in all_results[1:]:
        results_df_usgs_combined = results_df_usgs_combined.merge(
            component_results, 
            on=['Shale_play', 'Year'], 
            how='outer', 
            suffixes=('', '_dup')
        )
        # Remove duplicate suffix columns for Gas and Oil
        if 'Gas_dup' in results_df_usgs_combined.columns:
            results_df_usgs_combined['Gas'] = results_df_usgs_combined[['Gas', 'Gas_dup']].sum(axis=1, skipna=True)
            results_df_usgs_combined.drop(columns=['Gas_dup'], inplace=True)
        if 'Oil_dup' in results_df_usgs_combined.columns:
            results_df_usgs_combined['Oil'] = results_df_usgs_combined[['Oil', 'Oil_dup']].sum(axis=1, skipna=True)
            results_df_usgs_combined.drop(columns=['Oil_dup'], inplace=True)

    return results_df_usgs_combined


def return_results_df_ghgrp_combined_shales(result_dic, a, b, threshold, keep_nanapis, radius_factor):
    all_prod_indices_all = pd.read_csv(f'all_prod_indices_all_ghgrp.csv')
    all_prod_indices_all['X'] = (all_prod_indices_all['X']).round(1)
    all_prod_indices_all['Y'] = (all_prod_indices_all['Y']).round(1)

    all_prod_indices_all['Prod_X_grid'] = all_prod_indices_all['Prod_X_grid'].round(1)
    all_prod_indices_all['Prod_Y_grid'] = all_prod_indices_all['Prod_Y_grid'].round(1)
    
    print('Created all_prod_indices_all...')
    data_source = 'ghgrp'
    sample = 2
    radius_factor = 0.15
    result_ghgrp = result_dic[(threshold, a, b)]
    result_ghgrp['X'] = (result_ghgrp['X']).round(1)
    result_ghgrp['Y'] = (result_ghgrp['Y']).round(1)
    result_ghgrp = gpd.GeoDataFrame(
    result_ghgrp, 
    geometry=gpd.points_from_xy(result_ghgrp.X, result_ghgrp.Y)
    )
    result_ghgrp = gpd.sjoin(result_ghgrp, haynesville_shale_gdf, how="inner", predicate="within")
    
    
    components = ['CO2', 'C1']

    rename_mean = {comp: f"{comp}_mean" for comp in components}
    rename_std = {f"std_{comp}": f"std_{comp}_mean" for comp in components}

    result_ghgrp = result_ghgrp.rename(columns={**rename_mean, **rename_std})    
    
    print('Loaded results...')

    list_result_all_sgs = []
    for i in range(1, 4):
        a = 2
        b = 2
        interp_method = 'sgs_' + str(i)
        'Uploading sgs...'
        result_all_sgs = upload_result_simple(
                                   f'result_data_source_{data_source}_a_{a}_b_{b}_basin_*_sample_{sample}_interp_{interp_method}_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')
        
        result_all_sgs = gpd.GeoDataFrame(
        result_all_sgs, 
        geometry=gpd.points_from_xy(result_all_sgs.X, result_all_sgs.Y)
        )
        result_all_sgs = gpd.sjoin(result_all_sgs, haynesville_shale_gdf, how="inner", predicate="within")
        result_all_sgs = result_all_sgs.drop(columns=['std_C1'])
                
        
        base_cols = ['X', 'Y', 'T', 'C1_predic_gor', 'beta_C1']
        std_cols = [f"std_{comp}_mean" for comp in components]
        mean_cols = [f"{comp}_mean" for comp in components]

        subset = result_ghgrp[base_cols + mean_cols + std_cols]

        result_all_sgs = pd.merge(result_all_sgs,
                                  subset,
                                 on=['X','Y','T'])
        result_all_sgs = result_all_sgs[~pd.isna(result_all_sgs['C1_predic_gor'])]
        print('Adding predictions...')
        result_all_sgs['std_C1'] = result_all_sgs['std_C1_mean']

        result_all_sgs = result_all_sgs[~pd.isna(result_all_sgs['C1_predic_gor'])]
        result_all_sgs['C1'] = predic_beta(result_all_sgs['C1'], result_all_sgs['std_C1'], result_all_sgs['C1_predic_gor'], result_all_sgs['beta_C1'].iloc[0])
        list_result_all_sgs.append(result_all_sgs)

    new_list_result_all_sgs = []
    for result_all_sgs in list_result_all_sgs:
        print('Propagating uncertainties...')
        reference_date = '1918-08-19T00:00:00.000000000'
        result_all_sgs['Year'] = (pd.Timestamp(reference_date) + pd.to_timedelta(result_all_sgs['T'], unit='D')).dt.year
        def keep_closest_year(df, year_col='Year', group_cols=['X', 'Y'], target_year=2017):
            return df.loc[df.groupby(group_cols)[year_col].apply(lambda x: (x - target_year).abs().idxmin()).values]

        result_all_sgs = keep_closest_year(result_all_sgs.reset_index(drop=True), year_col='Year', group_cols=['X', 'Y'], target_year=2017)


        new_list_result_all_sgs.append(result_all_sgs)

    all_results = []
    components = ['CO2','C1']

    mean_cols = [f"{comp}_mean" for comp in components]

    for component in components + mean_cols:
        # compute weighted mean by year and component
        basin_year_means = compute_weighted_mean_by_year(
            new_list_result_all_sgs, value_column=component, weight_column='Gas', 
            basin_column='Shale_play', year_column='Year'
        )
        # compute overall mean across sgs --> CHANGE THAT BECAUSE FOR NOW ONLY MEAN, NOT WEIGHTED MEAN.
        component_results = compute_combined_results_shales(basin_year_means, component)
        all_results.append(component_results)

    # Merge results for all components into a single DataFrame
    results_df_ghgrp_combined = all_results[0]
    for component_results in all_results[1:]:
        results_df_ghgrp_combined = results_df_ghgrp_combined.merge(
            component_results, 
            on=['Shale_play', 'Year'], 
            how='outer', 
            suffixes=('', '_dup')
        )
        # Remove duplicate suffix columns for Gas and Oil
        if 'Gas_dup' in results_df_ghgrp_combined.columns:
            results_df_ghgrp_combined['Gas'] = results_df_ghgrp_combined[['Gas', 'Gas_dup']].sum(axis=1, skipna=True)
            results_df_ghgrp_combined.drop(columns=['Gas_dup'], inplace=True)
        if 'Oil_dup' in results_df_ghgrp_combined.columns:
            results_df_ghgrp_combined['Oil'] = results_df_ghgrp_combined[['Oil', 'Oil_dup']].sum(axis=1, skipna=True)
            results_df_ghgrp_combined.drop(columns=['Oil_dup'], inplace=True)

    return results_df_ghgrp_combined

# save average results by year by basin for figures
radius_factor = 0.175
C2_predicted = True
data_source = "usgs"  # replace with the appropriate value
filename = f"result_dic_r0175_{data_source}.pkl"
a = -1
b = 1
keep_nanapis = True
propagate = True
threshold = 1000

with open(filename, "rb") as f:
    result_dic = pickle.load(f)
    

# Initialize the dictionary with tqdm for progress monitoring
results_df_usgs_combined_dict_shales = {}
for (a, b) in tqdm(zip([-1, 1, -1, 1], [-1, 1, 1, -1]), desc="Pairs of (a, b)", total=4, position=0):
    for threshold in tqdm([1000], desc="Threshold", position=1, leave=False):
        for keep_nanapis in tqdm([True], desc="Keep NaNs", position=2, leave=False):
            for propagate in tqdm([True], desc="Propagate", position=3, leave=False):
                results_df_usgs_combined_dict_shales[(a, b, threshold, keep_nanapis, propagate)] = return_results_df_usgs_combined_shales(
                   result_dic, a, b, threshold, keep_nanapis, radius_factor, propagate, C2_predicted
                )
with open("results_df_usgs_combined_dict_C2_predicted_shales.pkl", "wb") as f:
    pickle.dump(results_df_usgs_combined_dict_shales, f)

# save average results by year by basin for figures
data_source = "ghgrp"  # replace with the appropriate value
filename = f"result_dic_r0175_{data_source}.pkl"

with open(filename, "rb") as f:
    result_dic = pickle.load(f)
radius_factor = 0.175
# Initialize the dictionary with tqdm for progress monitoring
results_df_ghgrp_combined_dict_shales = {}
for (a, b) in tqdm(zip([-1, 1, -1, 1], [-1, 1, 1, -1]), desc="Pairs of (a, b)", total=4, position=0):
    for threshold in tqdm([1000], desc="Threshold", position=1, leave=False):
        for keep_nanapis in tqdm([True], desc="Keep NaNs", position=2, leave=False):
            print((a,b,threshold))
            results_df_ghgrp_combined_dict_shales[(a, b, threshold, keep_nanapis)] = return_results_df_ghgrp_combined(
               result_dic, a, b, threshold, keep_nanapis, radius_factor
            )
with open("results_df_ghgrp_combined_dict_shales.pkl", "wb") as f:
    pickle.dump(results_df_ghgrp_combined_dict_shales, f)

ghgrp_production_gdf = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))  # all basins
raw_ghgrp_production_gdf = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_v2.csv'))  # all basins

# Define the function to plot comparison with additional raw data counts

def plot_comparison(threshold,
                    keep_nanapis,
                    component,
                    results_df_usgs_combined,
                    results_df_ghgrp_combined,
                    selected_basins,
                   ghgrp_production_gdf,
                    ghgrp_processing_gdf,
                    results_ghgrp_proc_by_basin,
                    year_usgs,
                    year_ghgrp,
                    appalachian_combined=False
                   ):
    
    
    ghgrp_production_gdf = gpd.GeoDataFrame(
    ghgrp_production_gdf, 
    geometry=gpd.points_from_xy(ghgrp_production_gdf.X, ghgrp_production_gdf.Y)
    )


    ghgrp_production_gdf_haynesville = gpd.sjoin(
    ghgrp_production_gdf, haynesville_shale_gdf, how="inner", predicate="within"
    )
    
    
    raw_data_usgs = filter_func(threshold, keep_nanapis).copy()
        
    raw_data_usgs = raw_data_usgs[~pd.isna(raw_data_usgs[component])]
    # Compute the Year column for raw data
    reference_date = '1918-08-19T00:00:00.000000000'
    raw_data_usgs['Year'] = (pd.Timestamp(reference_date) + pd.to_timedelta(raw_data_usgs['T'], unit='D')).dt.year
    raw_data_usgs.loc[raw_data_usgs['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'
    if appalachian_combined:
        raw_data_usgs['BASIN_NAME'] = raw_data_usgs['BASIN_NAME'].replace({
            'Appalachian Basin (EOA)': 'Appalachian Basin (combined)',
            'Appalachian Basin': 'Appalachian Basin (combined)'
        })
    
    
    raw_data_usgs = gpd.GeoDataFrame(
        raw_data_usgs, 
        geometry=gpd.points_from_xy(raw_data_usgs.X, raw_data_usgs.Y)
    ) 

    usgs_gdf_haynesville = gpd.sjoin(
        raw_data_usgs, haynesville_shale_gdf, how="inner", predicate="within"
    )

                                
    ghgrp_production_gdf_filtered = ghgrp_production_gdf[~pd.isna(ghgrp_production_gdf[component])].copy()
    ghgrp_production_gdf_filtered.loc[ghgrp_production_gdf_filtered['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'
    if appalachian_combined:
        ghgrp_production_gdf_filtered['BASIN_NAME'] = ghgrp_production_gdf_filtered['BASIN_NAME'].replace({
            'Appalachian Basin (EOA)': 'Appalachian Basin (combined)',
            'Appalachian Basin': 'Appalachian Basin (combined)'
        })

    
    unique_counts_df = ghgrp_production_gdf_filtered.groupby('BASIN_NAME').apply(
    lambda group: len(group[['FACILITY_ID', 'T']].drop_duplicates())
    ).reset_index(name='Unique_Counts').copy()
    
    
    df1 = results_df_usgs_combined[results_df_usgs_combined['BASIN_NAME'].isin(selected_basins)].copy()

    df2 = results_df_ghgrp_combined[results_df_ghgrp_combined['BASIN_NAME'].isin(selected_basins)].copy()

    
    def keep_closest_year_by_basin(df, year_col='Year', group_col='BASIN_NAME', target_year_dict=None):
        def get_closest_idx(group):
            basin = group.name
            if basin in target_year_dict:
                target = target_year_dict[basin]

                return (group[year_col] + 5 - target).abs().idxmin()
            else:
                return group[year_col].idxmax()  # fallback if basin not in dict

        return df.loc[df.groupby(group_col).apply(get_closest_idx).values]

    

    year_usgs_dict = (
        raw_data_usgs[raw_data_usgs[component].notna()]
        .groupby('BASIN_NAME')['Year']
        .max()
        .to_dict()
    )

    
    
#     year_ghgrp_dict = {
#             basin: 2022
#             for basin in ghgrp_production_gdf['BASIN_NAME'].unique()
#         }
    
    year_ghgrp_dict = (
        ghgrp_production_gdf[ghgrp_production_gdf[component].notna()]
        .groupby('BASIN_NAME')['Year']
        .max()
        .to_dict()
    )
    
    # USGS year
    haynesville_year_usgs = (
        usgs_gdf_haynesville[usgs_gdf_haynesville[component].notna()]['Year'].max()
    )
    year_usgs_dict['Haynesville-Bossier'] = haynesville_year_usgs

    # GHGRP year
    haynesville_year_ghgrp = (
        ghgrp_production_gdf_haynesville[ghgrp_production_gdf_haynesville[component].notna()]['Year'].max()
    )
    
    year_ghgrp_dict['Haynesville-Bossier'] = haynesville_year_ghgrp


    # If appalachian_combined is True, override with the min of both subregions
    if appalachian_combined:
        app_years_usgs = [year_usgs_dict.get('Appalachian Basin'), year_usgs_dict.get('Appalachian Basin (Eastern Overthrust Area)')]
        app_years_usgs = [y for y in app_years_usgs if y is not None]
        if app_years_usgs:
            year_usgs_dict['Appalachian Basin (combined)'] = min(app_years_usgs)

        app_years_ghgrp = [year_ghgrp_dict.get('Appalachian Basin'), year_ghgrp_dict.get('Appalachian Basin (Eastern Overthrust Area)')]
        app_years_ghgrp = [y for y in app_years_ghgrp if y is not None]
        if app_years_ghgrp:
            year_ghgrp_dict['Appalachian Basin (combined)'] = min(app_years_ghgrp)
    
    
    df1 = keep_closest_year_by_basin(
        df1.reset_index(drop=True),
        year_col='Year',
        group_col='BASIN_NAME',
        target_year_dict=year_usgs_dict
    )

    df2 = keep_closest_year_by_basin(
        df2.reset_index(drop=True),
        year_col='Year',
        group_col='BASIN_NAME',
        target_year_dict=year_ghgrp_dict
    )
    print(df2.Year.unique())
#     df1 = keep_closest_year(df1.reset_index(drop=True), year_col='Year', group_col='BASIN_NAME', target_year=year_usgs)
#     df2 = keep_closest_year(df2.reset_index(drop=True), year_col='Year', group_col='BASIN_NAME', target_year=year_ghgrp)
    

    df3 = results_ghgrp_proc_by_basin[results_ghgrp_proc_by_basin['BASIN_NAME'].isin(selected_basins)]

    # Add a 'Source' column to distinguish the dataframes
    df1['Source'] = 'USGS'
    df2['Source'] = 'GHGRP (production facilities)'
    df3['Source'] = 'GHGRP (processing facilities)'

    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Count raw data points by basin and source

    ghgrp_processing_gdf_filtered = ghgrp_processing_gdf[~pd.isna(ghgrp_processing_gdf[component])].copy()
    ghgrp_processing_gdf_filtered.loc[ghgrp_processing_gdf_filtered['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'

    if appalachian_combined:
        ghgrp_processing_gdf_filtered['BASIN_NAME'] = ghgrp_processing_gdf_filtered['BASIN_NAME'].replace({
            'Appalachian Basin (EOA)': 'Appalachian Basin (combined)',
            'Appalachian Basin': 'Appalachian Basin (combined)'
        })
    raw_data_ghgrp_proc = ghgrp_processing_gdf_filtered[ghgrp_processing_gdf_filtered['BASIN_NAME'].isin(selected_basins)].copy()
    
    raw_counts = {
        'USGS': raw_data_usgs.groupby('BASIN_NAME').size(),
        'GHGRP (production facilities)': unique_counts_df.set_index('BASIN_NAME')['Unique_Counts'],
        'GHGRP (processing facilities)': raw_data_ghgrp_proc.groupby('BASIN_NAME').size()
    }
    
    
    raw_counts['GHGRP (production facilities)']['Haynesville-Bossier'] = len(
        ghgrp_production_gdf_haynesville[['FACILITY_ID', 'T']].drop_duplicates()
    )

    raw_counts['USGS']['Haynesville-Bossier'] = usgs_gdf_haynesville.shape[0]
    
    
    raw_counts['GHGRP (processing facilities)']['Haynesville-Bossier'] = ghgrp_processing_joined_shales_gdf.shape[0]

    height = 9
    width = 18
    combined_df_copy = combined_df.copy()
    combined_df_copy[component] *= 100
    combined_df_copy[f'se_{component}'] *= 100
    fontsize = 28
    plt.figure(figsize=(width, height))

    colors = ['#A8DADC', '#457B9D', '#1D3557']  # Soft sky blue, bright aqua blue, and rich deep blue
    # Pivot the data for easier plotting
    pivot_df = combined_df_copy.pivot(index='BASIN_NAME', columns='Source', values=component)
    pivot_se_df = combined_df_copy.pivot(index='BASIN_NAME', columns='Source', values=f'se_{component}')


    # Filter the DataFrame for the selected basins
    pivot_df = pivot_df.loc[selected_basins]
    pivot_se_df = pivot_se_df.loc[selected_basins]
    
    # Ensure consistent order: USGS, GHGRP prod, GHGRP proc
    pivot_df = pivot_df[['USGS', 'GHGRP (production facilities)', 'GHGRP (processing facilities)']]
    pivot_se_df = pivot_se_df[['USGS', 'GHGRP (production facilities)', 'GHGRP (processing facilities)']]


    # Adjust standard errors to 95% confidence intervals
    ci_error_bars = pivot_se_df * 1.96  # Multiply standard errors by 1.96 to get 95% CI
    pivot_df = pivot_df.sort_index()
    pivot_se_df = pivot_se_df.sort_index()
    
    # Plotting the bar chart with error bars representing 95% confidence intervals
    ax = pivot_df.plot(kind='bar', yerr=ci_error_bars, capsize=5, width=0.7,  
                       error_kw={'elinewidth': 1.5, 'capthick': 1.5}, 
                       alpha=0.85, figsize=(width, height), color=colors)
    pivot_df = pivot_df.sort_index()
    # Adjust y-axis limits
    if component == 'C1':
        plt.ylim(0, 100)
    else:
        plt.ylim(0, 20)
    # Annotate the number of raw data points above each bar
    for i, basin in enumerate(pivot_df.index):
        for j, source in enumerate(pivot_df.columns):
            raw_count = raw_counts[source].get(basin, 0)
            ax.text(
            i + j * 0.25 - 0.3,  # Position above each bar
            ax.get_ylim()[1],  # Above bar + margin
            f'  {raw_count}',
            ha='left', va='bottom', fontsize=fontsize * 0.75, color='black', rotation=90, c='dimgray'
            )
    # Set x-tick labels
    # Set x-tick labels, removing '(combined)' for display
    xtick_labels = pivot_df.index.to_series().str.replace(' \(combined\)', '', regex=True)
    ax.set_xticks(range(len(pivot_df.index)))
    ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=fontsize)

    # Add a horizontal line at y = 90% for reference
    if component == 'C1':
        plt.axhline(y=90, color='gray', linestyle='--', linewidth=2, label='Reference (90%)')

    # Add labels and title with larger font sizes
    plt.xlabel('')
    plt.yticks(fontsize=fontsize)

    # Set y-ticks as percentages
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

    num_ticks = 5  # Adjust this as needed
    if component == 'C1':
        ax.set_yticks(np.concatenate([np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks), [90]]))
    else:
        ax.set_yticks(np.linspace(plt.ylim()[0], plt.ylim()[1], num_ticks))
    # Remove legend from the main plot
    
    ax.legend().set_visible(False)
    if component == 'C1':
        ax.set_ylabel(f'{component} (weighted avg.) [mol %]', fontsize=fontsize)
    else:
        ax.set_ylabel(r'CO$_2$ (weighted avg.) [mol %]', fontsize=fontsize)

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.savefig(f'figures_out/comparison_{component}_threshold_{threshold}_keep_nanapis_{keep_nanapis}.eps', format='eps', dpi=300)

    # Create a separate figure for the legend
    legend_fig, legend_ax = plt.subplots(figsize=(8, 4))
    handles, labels = ax.get_legend_handles_labels()
#     legend_ax.legend(handles, labels, title='Source', fontsize=fontsize, title_fontsize=fontsize, loc='center')
    legend_ax.legend(
        handles, 
        labels, 
        fontsize=fontsize, 
        loc='center', 
        bbox_to_anchor=(0.5, 0.5)  # Adjust these values to reposition the legend
    )

    legend_ax.axis('off')  # Remove axes from the legend figure
    legend_fig.savefig(f'figures_out/legend_{component}.eps', format='eps', dpi=300)
    plt.show()

    plt.close(legend_fig)
    return pivot_df, pivot_se_df

def add_combined_appalachian(data_source, df):
    df_appalachian = df[df.BASIN_NAME.isin(['Appalachian Basin', 'Appalachian Basin (EOA)'])]
    if data_source == 'usgs':
        value_cols = [
                'se_HE', 'Oil', 'Gas', 'CO2', 'se_CO2', 'H2', 'se_H2', 'N2', 'se_N2', 'H2S', 'se_H2S',
                'AR', 'se_AR', 'O2', 'se_O2', 'C1', 'se_C1', 'C1_mean', 'se_C1_mean', 'C2', 'se_C2',
                'C2_mean', 'se_C2_mean', 'C3', 'se_C3', 'N-C4', 'se_N-C4', 'I-C4', 'se_I-C4',
                'N-C5', 'se_N-C5', 'I-C5', 'se_I-C5', 'C6+', 'se_C6+'
            ]
    else:
         value_cols = [
            'Gas', 'CO2', 'se_CO2','C1', 'se_C1',
        ]
    # Separate by basin
    basins = df_appalachian['BASIN_NAME'].unique()
    df1 = df_appalachian[df_appalachian['BASIN_NAME'] == basins[0]].copy()
    df2 = df_appalachian[df_appalachian['BASIN_NAME'] == basins[1]].copy()

    # Result storage
    results = []

    # For each year in df1, find the closest year in df2
    for _, row1 in df1.iterrows():
        year1 = row1['Year']
        idx_closest = (df2['Year'] - year1).abs().idxmin()
        row2 = df2.loc[idx_closest]

        w1 = row1['Gas']
        w2 = row2['Gas']
        total_gas = w1 + w2

        weighted = {
            'Year': int(round((row1['Year'] + row2['Year']) / 2)),
            'BASIN_NAME': 'Appalachian Basin (combined)',
        }

        for col in value_cols:
            val1 = row1[col]
            val2 = row2[col]
            if pd.isna(val1) or pd.isna(val2):
                weighted[col] = np.nan
            else:
                weighted[col] = (w1 * val1 + w2 * val2) / total_gas

        results.append(weighted)
    df_weighted = pd.DataFrame(results)
    
    return pd.concat([df, df_weighted])

with open("results_df_usgs_combined_dict_C2_predicted_shales.pkl", "rb") as f:
    results_df_usgs_combined_dict_shales = pickle.load(f)
    
with open("results_df_ghgrp_combined_dict_shales.pkl", "rb") as f:
    results_df_ghgrp_combined_dict_shales = pickle.load(f)
    
    
df_usgs = results_df_usgs_combined_dict_shales[(-1, -1, 1000, True, True)]
df_ghgrp = results_df_ghgrp_combined_dict_shales[(-1, -1, 1000, True)]

df_usgs_shales = df_usgs.rename(columns={'Shale_play':'BASIN_NAME'})
df_ghgrp_shales = df_ghgrp.rename(columns={'Shale_play':'BASIN_NAME'})
results_ghgrp_proc_by_shale = results_ghgrp_proc_by_shale.rename(columns={'Shale_play':'BASIN_NAME'})

selected_basins = ['Anadarko Basin',
                          'Gulf Coast Basin (LA, TX)',
                          'San Joaquin Basin',
                          'Permian Basin',
                          'San Juan Basin',
                          'Appalachian Basin (EOA)',
                          'Appalachian Basin',

                          'Denver Basin',
                          'Fort Worth Syncline',
                          'Uinta Basin',
                          'Williston Basin',
                   'East Texas Basin',
                   'Arkla Basin',
                   'Haynesville-Bossier'
                          ]

a = 1
b = 1
threshold = 1000
component = 'C1'
with open("results_df_usgs_combined_dict_C2_predicted.pkl", "rb") as f:
    results_df_usgs_combined = pickle.load(f)[(a, b, threshold, True, True)]


with open("results_df_ghgrp_combined_dict.pkl", "rb") as f:
    results_df_ghgrp_combined = pickle.load(f)[(a, b, threshold, True)]
    
results_ghgrp_proc_by_basin.loc[results_ghgrp_proc_by_basin['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'
results_df_ghgrp_combined.loc[results_df_ghgrp_combined['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'
results_df_usgs_combined.loc[results_df_usgs_combined['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'
    
results_df_usgs_combined_copy = results_df_usgs_combined.copy()
results_df_ghgrp_combined_copy = results_df_ghgrp_combined.copy()
components = ['HE',  'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
for comp in components:
    results_df_usgs_combined_copy[comp] = results_df_usgs_combined_copy[comp + '_mean']

for comp in ['C1', 'CO2']:
    results_df_ghgrp_combined_copy[comp] = results_df_ghgrp_combined_copy[comp + '_mean']

results_df_usgs_combined_copy = add_combined_appalachian('usgs', results_df_usgs_combined_copy)
results_df_ghgrp_combined_copy = add_combined_appalachian('ghgrp', results_df_ghgrp_combined_copy)

results_ghgrp_proc_by_basin_copy = add_combined_appalachian('ghgrp',results_ghgrp_proc_by_basin.copy())


results_df_usgs_combined_copy = pd.concat([df_usgs_shales, results_df_usgs_combined_copy])

results_df_ghgrp_combined_copy = pd.concat([df_ghgrp_shales, results_df_ghgrp_combined_copy])

results_ghgrp_proc_by_basin_copy = pd.concat([results_ghgrp_proc_by_shale, results_ghgrp_proc_by_basin_copy])

for component in ['C1', 'CO2']:
    results_df_ghgrp_combined_copy[f'{component}'] /= 100
    results_df_ghgrp_combined_copy[f'se_{component}'] /= 100

pivot_df_c1, pivot_se_df = plot_comparison(threshold,
                                        True,
                                        'C1',
                                        results_df_usgs_combined_copy,
                                        results_df_ghgrp_combined_copy,
                                        selected_basins,
                                        ghgrp_production_gdf,
                                        ghgrp_processing_gdf,
                                        results_ghgrp_proc_by_basin_copy,
                                        2017,
                                        2017,
                                       False)

pivot_df, pivot_se_df = plot_comparison(threshold,
                                        True,
                                        'CO2',
                                        results_df_usgs_combined_copy,
                                        results_df_ghgrp_combined_copy,
                                        selected_basins,
                                        ghgrp_production_gdf,
                                        ghgrp_processing_gdf,
                                        results_ghgrp_proc_by_basin_copy,
                                        2017,
                                        2017,
                                       False)

def compute_cumulative_distribution(data, component, weight_column):
    """
    Computes the cumulative distribution for a single component, weighted by the specified weight column.

    Args:
        data (pd.DataFrame): The data containing component values and weights.
        component (str): The component to compute the distribution for.
        weight_column (str): The column representing the weights (e.g., 'Gas').

    Returns:
        tuple: Arrays for component values and cumulative distribution.
    """
    valid_data = data[[component, weight_column]].dropna()
    sorted_data = valid_data.sort_values(by=component)
    
    cumulative_weight = np.cumsum(sorted_data[weight_column])
    cumulative_weight /= cumulative_weight.iloc[-1]  # Normalize to [0, 1]
    
    # Add points at (0, 0) and (1, 1)
    values = sorted_data[component].values
    cdf = cumulative_weight.values

    # Prepend (0, 0) and append (max_value, 1)
    values = np.concatenate([[0], values, [values[-1]]])  # Include the max value for 1
    cdf = np.concatenate([[0], cdf, [1]])

    return values, cdf

def plot_cumulative_distributions_by_basin(data_sources, component, weight_column, title, selected_basins):
    """
    Plots cumulative distributions by basin for the specified component.

    Args:
        data_sources (dict): Dictionary with keys as labels and values as DataFrames.
        component (str): The component to plot.
        weight_column (str): The column representing the weights (e.g., 'Gas').
        title (str): Title for the plot.
    """

    colors = {'USGS': '#A8DADC', 'GHGRP (production facilities)': '#457B9D', 'GHGRP (processing facilities)': '#1D3557'}
    fontsize = 28
    
    for basin in sorted(selected_basins):
        plt.figure(figsize=(11.32, 8.11))
        for label, data in data_sources.items():
            # Filter data for the current basin
            basin_data = data[data['BASIN_NAME'] == basin]
            if not basin_data[~pd.isna(basin_data[component])].empty:
                values, cdf = compute_cumulative_distribution(basin_data, component, weight_column)
                plt.step(values, cdf, label=label, color=colors[label], linewidth=3, where='post')
        
        plt.xlabel(f'{component} fraction [mol %]', fontsize=fontsize)
        plt.ylabel('Cumulative fraction', fontsize=fontsize)
        plt.title(f'{basin}', fontsize=fontsize)
        if basin == 'Anadarko Basin':
            plt.legend(fontsize=fontsize - 2)
        plt.grid(True, linestyle='--')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
#         plt.tight_layout()
        plt.xticks(ticks=plt.xticks()[0], labels=[f'{tick:.0f}%' for tick in plt.xticks()[0]], fontsize=fontsize)
        plt.xlim(0,100)
        plt.savefig(f'figures_out/cdf_{component}_basin_{basin}.eps', format='eps', dpi=300)
        plt.show()
        if basin == 'Anadarko Basin':
            legend = plt.legend(fontsize=fontsize - 2)

            # Save the legend as a separate figure
            fig_legend = plt.figure(figsize=(6, 2))
            ax_legend = fig_legend.add_subplot(111)
            ax_legend.axis('off')
            fig_legend.legend(*legend.axes.get_legend_handles_labels(),
                              loc='center', ncol=3, fontsize=fontsize - 2, frameon=False)
            fig_legend.savefig(f'figures_out/legend_{component}.eps', format='eps', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig_legend)
            
threshold = 1000
a = 1
b = 1


data_source = "usgs"  # replace with the appropriate value
filename = f"result_dic_r0175_{data_source}.pkl"

with open(filename, "rb") as f:
    usgs_data = pickle.load(f)[(threshold, a, b)].rename(columns={'Monthly Gas':'Gas',
                                                                             'Monthly Oil':'Oil'})

data_source = "ghgrp"  # replace with the appropriate value
filename = f"result_dic_r0175_{data_source}.pkl"

with open(filename, "rb") as f:
    ghgrp_production_data = pickle.load(f)[(threshold, a, b)].rename(columns={'Monthly Gas':'Gas',
                                                                             'Monthly Oil':'Oil'})

ghgrp_processing_data = ghgrp_processing_gdf


def keep_closest_year_by_xy_using_basin(df, year_col='Year', group_cols=['X', 'Y'], target_year_dict=None):
    """
    For each (X, Y) location, keep the row where Year is closest to the target year 
    defined by the basin in target_year_dict, only if within 10 years.
    """

    def get_closest_idx(group):
        basin = group['BASIN_NAME'].iloc[0]
        if basin in target_year_dict:
            target = target_year_dict[basin]
            adjusted_diff = (group[year_col] + 5 - target).abs()
            min_diff = adjusted_diff.min()
            if min_diff <= 5:
                return adjusted_diff.idxmin()
        return None  # exclude group if no valid year found

    closest_indices = df.groupby(group_cols).apply(get_closest_idx).dropna().astype(int)
    return df.loc[closest_indices.values]

raw_data_usgs = filter_func(threshold, keep_nanapis).copy()

raw_data_usgs = raw_data_usgs[~pd.isna(raw_data_usgs['C1'])]

year_usgs_dict = (
raw_data_usgs[raw_data_usgs['C1'].notna()]
.groupby('BASIN_NAME')['Year']
.max()
.to_dict()
)


year_ghgrp_dict = {
        basin: 2022
        for basin in ghgrp_production_gdf['BASIN_NAME'].unique()
    }

year_ghgrp_dict = (
    ghgrp_production_gdf[ghgrp_production_gdf[component].notna()]
    .groupby('BASIN_NAME')['Year']
    .max()
    .to_dict()
)

usgs_data.loc[usgs_data['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'
ghgrp_production_data.loc[ghgrp_production_data['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'
ghgrp_processing_data.loc[ghgrp_processing_data['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'


ghgrp_production_data = keep_closest_year_by_xy_using_basin(
    df=ghgrp_production_data,
    year_col='Year',
    group_cols=['X', 'Y'],
    target_year_dict=year_ghgrp_dict
)

usgs_data = keep_closest_year_by_xy_using_basin(
    df=usgs_data,
    year_col='Year',
    group_cols=['X', 'Y'],
    target_year_dict=year_usgs_dict
)

ghgrp_processing_data = ghgrp_processing_gdf

# Prepare data sources
data_sources_c1 = {
    'USGS': usgs_data,
    'GHGRP (production facilities)': ghgrp_production_data,
    'GHGRP (processing facilities)': ghgrp_processing_data
}

data_sources_co2 = {
    'USGS': usgs_data,
    'GHGRP (production facilities)': ghgrp_production_data,
    'GHGRP (processing facilities)': ghgrp_processing_data
}

# Plot cumulative distributions for C1 by basin
plot_cumulative_distributions_by_basin(data_sources_c1, 'C1', 'Gas',
                                        'Cumulative Distribution of C1 by Basin', selected_basins)


# Define the function to plot all components with smooth error bars
def plot_all_components_evolution(results_df_dict, thresholds, keep_nanapis, components, a, b):
    height = 30
    width = 30
    fontsize = 24  # Increased font size for paper quality

    # Sort thresholds to ensure smooth linking from smallest to largest
    thresholds = sorted(thresholds)

    # Determine subplot grid size
    n_components = len(components)
    n_cols = 3
    n_rows = (n_components + n_cols - 1) // n_cols

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), sharex=True)
    axes = axes.flatten()
    
    for idx, component in enumerate(components):
        evolution_data = []
        for threshold in thresholds:
            results_df = results_df_dict[(a, b, threshold, keep_nanapis, True)]
            mean_component = f"{component}_mean"  # Fixed this line

            raw_data_usgs = filter_func(threshold, keep_nanapis).copy()

            
            def keep_closest_year_by_basin(df, year_col='Year', group_col='BASIN_NAME', target_year_dict=None):
                def get_closest_idx(group):
                    basin = group.name
                    if basin in target_year_dict:
                        target = target_year_dict[basin]
                        return (group[year_col] + 5 - target).abs().idxmin()
                    else:
                        return group[year_col].idxmax()  # fallback if basin not in dict

                return df.loc[df.groupby(group_col).apply(get_closest_idx).values]



            year_usgs_dict = (
                raw_data_usgs[raw_data_usgs[component].notna()]
                .groupby('BASIN_NAME')['Year']
                .max()
                .to_dict()
            )

            results_df = keep_closest_year_by_basin(
                results_df.reset_index(drop=True),
                year_col='Year',
                group_col='BASIN_NAME',
                target_year_dict=year_usgs_dict
            )


            weighted_mean = (results_df[mean_component] * results_df['Gas']).sum() / results_df['Gas'].sum()
            weighted_se = (results_df[f'se_{component}'] * results_df['Gas']).sum() / results_df['Gas'].sum()

            if threshold == 1000:
                threshold = 100
            
            evolution_data.append({'Threshold': threshold, 'Mean': weighted_mean, 'SE': weighted_se})


        # Convert the evolution data to a DataFrame
        evolution_df = pd.DataFrame(evolution_data)

        # Sort the DataFrame by threshold to ensure correct linking order
        evolution_df = evolution_df.sort_values(by='Threshold')

        # Multiply by 100 for percentage representation
        evolution_df['Mean'] *= 100
        evolution_df['SE'] *= 100

        # Plot the evolution of the component
        ax = axes[idx]
        ax.plot(evolution_df['Threshold'], evolution_df['Mean'], '-o', label=component, linewidth=2, markersize=8)

        # Add smooth error bars with lighter color for compatibility with vector editors
        lighter_color = sns.light_palette("blue", n_colors=1)[0]
        ax.fill_between(
            evolution_df['Threshold'], 
            evolution_df['Mean'] - evolution_df['SE'] * 1.96, 
            evolution_df['Mean'] + evolution_df['SE'] * 1.96, 
            color=lighter_color, edgecolor='none'
        )

        # Format the subplot
#         if component == 'C1':
#             ax.axhline(y=90, color='gray', linestyle='--', linewidth=2, label='Reference (90%)')
        # Do not set individual x-axis labels
        if idx >= (n_components - n_cols):  # Optional: only show ticks on bottom row
            ax.tick_params(axis='x', labelsize=fontsize)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(f'{component} (%)', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}%'))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax.legend(fontsize=fontsize, loc='upper right')

    # Remove unused subplots
    for idx in range(len(components), len(axes)):
        fig.delaxes(axes[idx])

    fig.supxlabel('Threshold', fontsize=fontsize + 2)
    plt.tight_layout()

    plt.savefig(f'figures_out/all_components_evolution_with_threshold_keep_nanapis_{keep_nanapis}.eps', format='eps', dpi=300)
    plt.show()

# Example usage
a = 1
b = 1

thresholds = [1000, 10, 25, 50]
keep_nanapis = True
components = ['C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+', 'HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']

with open("results_df_usgs_combined_dict_C2_predicted_all_sgs.pkl", "rb") as f:
    results_df_usgs_combined_dict_C2_predicted = pickle.load(f)
    
plot_all_components_evolution(results_df_usgs_combined_dict_C2_predicted, thresholds, keep_nanapis, components, a, b)


with open("results_df_usgs_combined_dict_C2_predicted.pkl", "rb") as f:
    results_df_usgs_combined_dict = pickle.load(f)

    
with open("results_df_ghgrp_combined_dict.pkl", "rb") as f:
    results_df_ghgrp_combined_dict = pickle.load(f)

fontsize = 24
components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
threshold = 1000
keep_nanapis = True
a = 1
b = 1

results_df_usgs_combined = results_df_usgs_combined_dict[(1, 1, 1000, True, True)]

results_df_ghgrp_combined = results_df_ghgrp_combined_dict[(1, 1, 1000, True)]

for comp in components:
    results_df_usgs_combined[comp] = results_df_usgs_combined[comp + '_mean']

for comp in ['C1', 'CO2']:
    results_df_ghgrp_combined[comp] = results_df_ghgrp_combined[comp + '_mean']

def keep_closest_year(df, year_col='Year', group_col='BASIN_NAME', target_year=2017):
    return df.loc[df.groupby(group_col)[year_col].apply(lambda x: (x - target_year).abs().idxmin()).values]


raw_data_usgs = filter_func(threshold, keep_nanapis).copy()


def keep_closest_year_by_basin(df, year_col='Year', group_col='BASIN_NAME', target_year_dict=None):
    def get_closest_idx(group):
        basin = group.name
        if basin in target_year_dict:
            target = target_year_dict[basin]
            return (group[year_col] +5- target).abs().idxmin()
        else:
            return group[year_col].idxmax()  # fallback if basin not in dict

    return df.loc[df.groupby(group_col).apply(get_closest_idx).values]

# year_ghgrp_dict = (
#     ghgrp_production_gdf[ghgrp_production_gdf[component].notna()]
#     .groupby('BASIN_NAME')['Year']
#     .max()
#     .to_dict()
# )

# year_usgs_dict = (
#     raw_data_usgs[raw_data_usgs[component].notna()]
#     .groupby('BASIN_NAME')['Year']
#     .max()
#     .to_dict()
# )

from collections import defaultdict

def collect_closest_year_rows(df, components, source_name, year_dict_all):
    rows = []
    for component in components:
        if component not in df.columns:
            continue  # skip if component not present

        df_comp = df[df[component].notna()].copy()
        year_dict = year_dict_all[component] 

        df_closest = keep_closest_year_by_basin(
            df_comp.reset_index(drop=True),
            year_col='Year',
            group_col='BASIN_NAME',
            target_year_dict=year_dict
        )
        rows.append(df_closest)
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=['BASIN_NAME'])  # keep one row per basin
    combined['Source'] = source_name
    return combined



year_usgs_dict_all = {
    comp: {k: v - 5 for k, v in raw_data_usgs[raw_data_usgs[comp].notna()]
           .groupby('BASIN_NAME')['Year'].max().to_dict().items()}
    for comp in components
}



year_ghgrp_dict_all = {
    comp: {k: 2022 for k, v in ghgrp_production_gdf[ghgrp_production_gdf[comp].notna()]
           .groupby('BASIN_NAME')['Year'].max().to_dict().items()}
    for comp in ['C1', 'CO2']
}

year_ghgrp_dict_all = {
    comp: {k: v for k, v in ghgrp_production_gdf[ghgrp_production_gdf[comp].notna()]
           .groupby('BASIN_NAME')['Year'].max().to_dict().items()}
    for comp in components
}


results_df_usgs_combined = collect_closest_year_rows(
    results_df_usgs_combined,
    components,
    'USGS',
    year_usgs_dict_all
)

results_df_ghgrp_combined = collect_closest_year_rows(
    results_df_ghgrp_combined,
    ['C1', 'CO2'],  # GHGRP only has these
    'GHGRP',
    year_ghgrp_dict_all
)



# Apply to both DataFrames
# results_df_usgs_combined = keep_closest_year(results_df_usgs_combined, year_col='Year', group_col='BASIN_NAME', target_year=2017)
# results_df_ghgrp_combined = keep_closest_year(results_df_ghgrp_combined)

df1 = results_df_usgs_combined.copy()

df2 = results_df_ghgrp_combined.copy()

df1.loc[df1['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'
df2.loc[df2['BASIN_NAME'] == 'Appalachian Basin (Eastern Overthrust Area)', 'BASIN_NAME'] = 'Appalachian Basin (EOA)'

df1 = add_combined_appalachian('usgs', df1.copy())
df2 = add_combined_appalachian('ghgrp', df2.copy())
selected_basins = ['Anadarko Basin',
                          'Gulf Coast Basin (LA, TX)',
                          'San Joaquin Basin',
                          'Permian Basin',
                          'San Juan Basin',
                          'Appalachian Basin (EOA)',

                          'Denver Basin',
                          'Fort Worth Syncline',
                          'Uinta Basin',
                          'Williston Basin',
                   'East Texas Basin',
                   'Arkla Basin'
                          ]
df1 = df1[df1['BASIN_NAME'].isin(selected_basins)].copy()
df2 = df2[df2['BASIN_NAME'].isin(selected_basins)].copy()

df1 = df1.fillna(0)


for component in ['C1', 'CO2']:
    df2[f'{component}'] /= 100
    df2[f'se_{component}'] /= 100
    

# Assuming components is predefined, and df1, df2 are the datasets

# Merge the two DataFrames on 'BASIN_NAME', ignoring YEARS
merged_df = pd.merge(df1.drop(columns=['YEARS'], errors='ignore'), 
                     df2[['BASIN_NAME', 'C1', 'se_C1', 'CO2', 'se_CO2']].drop(columns=['YEARS'], errors='ignore'), 
                     on='BASIN_NAME', how='inner')
# merged_df = pd.merge(df1.drop(columns=['YEARS'], errors='ignore'), 
#                      df2[['BASIN_NAME', 'C1', 'se_C1']].drop(columns=['YEARS'], errors='ignore'), 
#                      on='BASIN_NAME', how='inner')

# # Replace the C1 and se_C1 values in df1 with those from df2
merged_df['C1'] = merged_df['C1_y']
merged_df['se_C1'] = merged_df['se_C1_y']
merged_df['CO2'] = merged_df['CO2_y']
merged_df['se_CO2'] = merged_df['se_CO2_y']

# # Drop duplicate columns resulting from themerge
merged_df.drop(columns=['C1_x', 'se_C1_x', 'C1_y', 'se_C1_y'], inplace=True)
merged_df.drop(columns=['CO2_x', 'se_CO2_x', 'CO2_y', 'se_CO2_y'], inplace=True)

# Normalize all component values and propagate uncertainties
merged_df['total'] = merged_df[components].sum(axis=1)
merged_df['total_uncertainty'] = np.sqrt(
    sum((merged_df[f'se_{component}'] ** 2 for component in components))
)

# Create new DataFrame to store normalized values and uncertainties
normalized_data = pd.DataFrame()
for component in components:
    # Normalize values
    normalized_data[component] = merged_df[component] * 100 / merged_df['total']
    # Propagate uncertainties
    normalized_data[f'se_{component}'] = normalized_data[component] * np.sqrt(
        (merged_df[f'se_{component}'] / merged_df[component]) ** 2 +
        (merged_df['total_uncertainty'] / merged_df['total']) ** 2
    )

    
    
components_wo_C1_CO2 = [comp for comp in components if comp not in ['C1', 'CO2']]
# components_wo_C1_CO2 = [comp for comp in components if comp not in ['C1']]

# Total of non-C1 and non-CO2 components
merged_df['total_wo_C1_CO2'] = merged_df[components_wo_C1_CO2].sum(axis=1)
merged_df['remaining_share'] = 100 - (merged_df['C1'] + merged_df['CO2']) * 100  # both in fraction
merged_df['remaining_share'] = 100 - (merged_df['C1']) * 100  # both in fraction

# Total uncertainty (excluding C1 and CO2)
merged_df['total_uncertainty_wo_C1_CO2'] = np.sqrt(
    sum((merged_df[f'se_{component}'] ** 2 for component in components_wo_C1_CO2))
)

# Create new DataFrame to store normalized values and uncertainties
normalized_data = pd.DataFrame(index=merged_df.index)
normalized_data['C1'] = merged_df['C1'] * 100  # Convert to percent
normalized_data['se_C1'] = merged_df['se_C1'] * 100
normalized_data['CO2'] = merged_df['CO2'] * 100
normalized_data['se_CO2'] = merged_df['se_CO2'] * 100

for component in components_wo_C1_CO2:
    # Normalize values to remaining share
    normalized_data[component] = merged_df[component] * merged_df['remaining_share'] / merged_df['total_wo_C1_CO2']
    
    # Propagate uncertainties
    normalized_data[f'se_{component}'] = normalized_data[component] * np.sqrt(
        (merged_df[f'se_{component}'] / merged_df[component]) ** 2 +
        (merged_df['total_uncertainty_wo_C1_CO2'] / merged_df['total_wo_C1_CO2']) ** 2
    )


# Sort components based on quantities in the Anadarko Basin
anadarko_data = normalized_data.loc[merged_df['BASIN_NAME'] == 'Anadarko Basin']
component_order = anadarko_data[components].mean().sort_values(ascending=False).index.tolist()

# Prepare data for heatmap
heatmap_data = normalized_data[components].set_index(merged_df['BASIN_NAME'])[component_order]
heatmap_data = heatmap_data.T
# Save uncertainties for future use
uncertainties_data = normalized_data[[f'se_{component}' for component in components]].set_index(merged_df['BASIN_NAME'])

vmin = np.percentile(heatmap_data.values.flatten(), 2)
vmax = np.percentile(heatmap_data.values.flatten(), 98)

# Create the heatmap with specified vmin and vmax
plt.figure(figsize=(24, 12))
ax = sns.heatmap(
    heatmap_data,
    cmap=sns.cubehelix_palette(start=2.6, hue=1.2, dark=0.3, light=0.9, gamma=1, reverse=False, as_cmap=False, rot=0),
    annot=True,
    fmt=".2f",
    annot_kws={'fontsize': fontsize},  # Set font size for annotations
    cbar_kws={'label': 'Normalized Component Values', 'shrink': 0.8},
    vmin=vmin,  # Set the minimum value to the 4th percentile
    vmax=vmax,  # Set the maximum value to the 95th percentile
    linewidths=0.5
)

# Customize the colorbar
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=fontsize - 2)
def format_as_percentage(x, pos):
    return f"{x:.0f}%"  # Format as integer with a '%' sign
cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_as_percentage))
cbar.set_label(None)

# Customize labels for academic paper styling
plt.xticks(fontsize=fontsize, rotation=45, ha='right')
plt.yticks(fontsize=fontsize, rotation=0)
plt.xlabel(None)
plt.ylabel(None)

# Adjust layout for tight appearance
plt.tight_layout()

# Save the figure
plt.savefig('figures_out/heatmap_fullgascomposition.eps', format='eps', dpi=300)
plt.savefig('figures_out/heatmap_fullgascomposition.png', format='eps', dpi=300)

plt.show()

# The final DataFrame containing uncertainties is `uncertainties_data`

# Add Gas back to the normalized data for weighting
normalized_data['Gas'] = merged_df['Gas']

# weighted_means = {}
weighted_ses = {}
weighted_stds = {}
weighted_means = {}
for comp in components:
    values = normalized_data[comp]
    ses = normalized_data[f'se_{comp}']
    weights = normalized_data['Gas']
    norm_weights = weights / weights.sum()

    # Weighted mean
    weighted_mean = np.average(values, weights=weights)

    # Weighted standard error (uncertainty on the mean)
    weighted_se = np.sqrt(np.average(ses**2, weights=weights))

    # Weighted standard deviation (spread across basins)
    weighted_std = np.sqrt(np.average((values - weighted_mean)**2, weights=weights))

    weighted_means[comp] = weighted_mean
    weighted_ses[f'se_{comp}'] = weighted_se
    weighted_stds[f'std_{comp}'] = weighted_std

# Combine into a DataFrame
national_average_df = pd.DataFrame({
    'Component': components,
    'Weighted Mean (%)': [weighted_means[c] for c in components],
    'Standard Error (%)': [weighted_ses[f'se_{c}'] for c in components],
    'Std Across Basins (%)': [weighted_stds[f'std_{c}'] for c in components]
})


# results from Evan updated

heatmap_data = heatmap_data.T
    
merged_basin = pd.merge(uncertainties_data, heatmap_data, left_index=True, right_index=True, how='inner')  # 'inner' keeps only matching rows

data = {
    'Scenario': ['Kairos Permian 18-20',
                 'CM Permian 19',
                 'CM Permian 20',
                 'CM Permian Sum 21',
                'CM Permian Fall 21',
                 'CM San Joaquin 16',
                 'CM San Joaquin 17',
                 'CM San Joaquin 20 Summer',
                'CM San Joaquin 20 Fall',
                 'CM San Joaquin 21',
                 'CM DJ Sum 21',
                'CM DJ Fall 21',
                 'CM Pennsylvania 21',
                 'CM Uinta 20',
                 'Kairos Fort Worth 21'],
    'Year':[[2018,2019,2020],
            [2019],
            [2020],
            [2021],
            [2021],
            [2016],
            [2017],
            [2020],
            [2020],
            [2021],
            [2021],
            [2021],
            [2021],
            [2020],
            [2021]],
            

    'BASIN_NAME':['Permian Basin',
             'Permian Basin',
             'Permian Basin',
             'Permian Basin',
             'Permian Basin',
             'San Joaquin Basin',
             'San Joaquin Basin',
             'San Joaquin Basin',
             'San Joaquin Basin',
             'San Joaquin Basin',
             'Denver Basin',
             'Denver Basin',
             'Appalachian Basin (EOA)',
             'Uinta Basin',
             'Fort Worth Syncline'],
    'Methane_emissions': ['188 [176, 203]',
                          '557 [534, 582]',
                          '134 [124, 145]',
                          '130 [122, 139]',
                          '131 [121, 142]',
                         '11 [10, 13]',
                          '16 [14, 18]',
                          '11 [10, 12]',
                          '9 [8, 11]',
                          '5 [4, 6]',
                          '21 [20, 24]',
                         '23 [21, 26]',
                          '76 [67, 87]',
                          '20 [19, 22]',
                          '22 [20, 24]'],
    
    'Methane_production': ['1,952',
                           '10,527',
                           '4,767',
                           '5,950',
                           '6,228',
                           '265',
                           '285',
                           '231',
                           '245',
                           '198',
                           '1,997',
                           '1,973',
                           '10,242',
                           '351',
                           '651'],
    'Methane_loss_rate': ['9.63 [9.04, 10.39]',
                          '5.29 [5.08, 5.53]',
                          '2.81 [2.6, 3.04]',
                          '2.19 [2.05, 2.34]',
                          '2.1 [1.94, 2.27]',
                         '4.23 [3.63, 4.92]',
                          '5.64 [5.04, 6.34]',
                          '4.63 [4.14, 5.23]',
                          '3.79 [3.26, 4.42]',
                          '2.54 [2.2, 2.91]',
                         '1.08 [0.98, 1.18]',
                          '1.17 [1.05, 1.33]',
                          '0.75 [0.65, 0.84]',
                          '5.74 [5.3, 6.21]',
                          '3.33 [3.01, 3.66]']}


emissions_df = pd.DataFrame(data)
# Remove the numbers in brackets using regular expressions
emissions_df['Methane_production'] = emissions_df['Methane_production'].str.replace(',', '')


# Extract lower and upper bounds for Methane emissions
emissions_df['Methane_emissions_median'] = emissions_df['Methane_emissions'].str.extract(r'(\d+) \[(\d+), (\d+)]')[0].astype(int)
emissions_df['Methane_emissions_lower'] = emissions_df['Methane_emissions'].str.extract(r'(\d+) \[(\d+), (\d+)]')[1].astype(int)
emissions_df['Methane_emissions_upper'] = emissions_df['Methane_emissions'].str.extract(r'(\d+) \[(\d+), (\d+)]')[2].astype(int)

# Extract lower and upper bounds for Methane loss rate
emissions_df['Methane_loss_rate_median_original'] = emissions_df['Methane_loss_rate'].str.extract(r'(\d+\.\d+) \[(\d+\.\d+), (\d+\.\d+)]')[0].astype(float)
emissions_df['Methane_loss_rate_lower_original'] = emissions_df['Methane_loss_rate'].str.extract(r'(\d+\.\d+) \[(\d+\.\d+), (\d+\.\d+)]')[1].astype(float)
emissions_df['Methane_loss_rate_upper_original'] = emissions_df['Methane_loss_rate'].str.extract(r'(\d+\.\d+) \[(\d+\.\d+), (\d+\.\d+)]')[2].astype(float)
emissions_df.drop(columns=['Methane_emissions', 'Methane_loss_rate'], inplace=True)

emissions_df['Methane_production'] = emissions_df['Methane_production'].astype(int)

# Define P (initially 0.9, for 90%)
P_initial = 0.9

# Compute loss rate using the provided formula
emissions_df['Methane_loss_rate_median'] = 100 * emissions_df['Methane_emissions_median'] / (emissions_df['Methane_production'] * P_initial)
emissions_df['Methane_loss_rate_lower'] = 100 * emissions_df['Methane_emissions_lower'] / (emissions_df['Methane_production'] * P_initial)
emissions_df['Methane_loss_rate_upper'] = 100 * emissions_df['Methane_emissions_upper'] / (emissions_df['Methane_production'] * P_initial)

from matplotlib.offsetbox import AnnotationBbox, TextArea, VPacker

fontsize = 40

# Step 1: Compute Fractional Loss and Add BASIN_NAME
def compute_fractional_loss_with_uncertainties(emissions_df, merged_basin):
    # Ensure numeric types
    emissions_df['Methane_production'] = emissions_df['Methane_production'].astype(float)
    emissions_df['Methane_emissions_median'] = emissions_df['Methane_emissions_median'].astype(float)
    merged_basin['C1'] = merged_basin['C1'].astype(float)
    merged_basin['se_C1'] = merged_basin['se_C1'].astype(float)

    combined_df = pd.merge(emissions_df, merged_basin, on='BASIN_NAME', how='inner')

    # Compute fractional loss (AFTER) using updated methane fraction
    combined_df['Fractional_loss_after'] = (
        combined_df['Methane_emissions_median'] /
        (combined_df['C1'] / 100 * combined_df['Methane_production'])  # Convert C1 from percentage
    ) * 100  # Convert to percentage

    # Convert uncertainties in methane fraction (C1) to 95% confidence intervals
    combined_df['C1_conf_interval'] = combined_df['se_C1'] * 1.96  # Standard errors to 95% CI

    # Compute uncertainty for the "after" fractional loss
    combined_df['Fractional_loss_after_uncertainty'] = combined_df['Fractional_loss_after'] * np.sqrt(
        ((combined_df['Methane_emissions_upper'] - combined_df['Methane_emissions_lower']) / 
         (2 * combined_df['Methane_emissions_median'])) ** 2 +  # Emissions uncertainty
        (combined_df['C1_conf_interval'] / combined_df['C1']) ** 2  # Methane fraction uncertainty
    )

    return combined_df

# Apply the function
updated_df = compute_fractional_loss_with_uncertainties(emissions_df, merged_basin)

# Step 2: Prepare data for plotting
before_data = updated_df[['Scenario', 'BASIN_NAME', 'Methane_loss_rate_median', 'Methane_loss_rate_lower', 'Methane_loss_rate_upper']].copy()
before_data.rename(
    columns={
        'Methane_loss_rate_median': 'Fractional_loss',
        'Methane_loss_rate_lower': 'Lower_bound',
        'Methane_loss_rate_upper': 'Upper_bound'
    }, inplace=True
)
before_data['Loss_Type'] = '90%'

after_data = updated_df[['Scenario', 'BASIN_NAME', 'Fractional_loss_after', 'Fractional_loss_after_uncertainty']].copy()
after_data.rename(
    columns={
        'Fractional_loss_after': 'Fractional_loss',
        'Fractional_loss_after_uncertainty': 'Error_bar'
    }, inplace=True
)
after_data['Lower_bound'] = after_data['Fractional_loss'] - after_data['Error_bar']
after_data['Upper_bound'] = after_data['Fractional_loss'] + after_data['Error_bar']
after_data['Loss_Type'] = 'Adjusted (ours)'

# Combine the datasets for side-by-side bars
combined_data = pd.concat([before_data, after_data])

# Step 3: Add gaps between groups based on BASIN_NAME
grouped_changes = {}
basins = updated_df['BASIN_NAME'].unique()  # Use BASIN_NAME directly

x_positions = []  # Adjusted x-positions for bars
gap_size = 3  # Define the gap between basin groups
current_position = 0
scenario_to_position = {}  # Map scenarios to x-positions

for basin in basins:
    basin_data = updated_df[updated_df['BASIN_NAME'] == basin]
    before_loss = before_data[before_data['Scenario'].isin(basin_data['Scenario'])]['Fractional_loss'].mean()
    after_loss = after_data[after_data['Scenario'].isin(basin_data['Scenario'])]['Fractional_loss'].mean()
    percentage_increase = ((after_loss - before_loss) / before_loss) * 100
    grouped_changes[basin] = percentage_increase


    # Assign x-positions for the scenarios in this basin
    basin_positions = list(range(current_position, current_position + len(basin_data)))
    x_positions.extend(basin_positions)
    
    # Map each scenario to its x-position
    for i, scenario in enumerate(basin_data['Scenario']):
        scenario_to_position[scenario] = basin_positions[i]
    
    current_position = basin_positions[-1] + gap_size  # Add a gap for the next basin

# Update `Scenario` as a categorical type with the correct order
updated_df['Scenario'] = pd.Categorical(
    updated_df['Scenario'], 
    categories=scenario_to_position.keys(), 
    ordered=True
)

# Create new x-axis positions with gaps for plotting
bar_positions = [scenario_to_position[scenario] for scenario in updated_df['Scenario']]
# Step 4: Plot the bars
plt.figure(figsize=(38, 20))
ax = plt.gca()
for i, scenario in enumerate(updated_df['Scenario']):
    before_loss = before_data[before_data['Scenario'] == scenario]['Fractional_loss'].values[0]
    after_loss = after_data[after_data['Scenario'] == scenario]['Fractional_loss'].values[0]
    before_err = before_data[before_data['Scenario'] == scenario]['Upper_bound'].values[0] - before_loss
    after_err = after_data[after_data['Scenario'] == scenario]['Upper_bound'].values[0] - after_loss
# colors = ['#FDD7D7', 'palevioletred']

    # Plot "Before" bar
    ax.bar(
        x=bar_positions[i] - 0.2,  # Offset for "90%" bar
        height=before_loss,
#         color='#aebbb7',
#         color = '#b8ad96',
        color='#d9c8b4',
        
        edgecolor='none',

        alpha=1,
        label='90%' if i == 0 else "",
        width=0.4
    )
#     colors = ['#A8DADC', '#457B9D', '#1D3557']  # Soft sky blue, bright aqua blue, and rich deep blue

    # Plot "After" bar
    ax.bar(
        x=bar_positions[i] + 0.2,  # Offset for "Adjusted (ours)" bar
        height=after_loss,
#         color= '#667874',
        color='#a77945',
#         color= '#b9a183',
        edgecolor='none',
        label='Adjusted (ours)' if i == 0 else "",
        width=0.4
    )

    # Add error bars
    ax.errorbar(
        x=bar_positions[i] - 0.2,
        y=before_loss,
        yerr=before_err,
        fmt='none',
        ecolor='black',
        capsize=5
    )
    ax.errorbar(
        x=bar_positions[i] + 0.2,
        y=after_loss,
        yerr=after_err,
        fmt='none',
        ecolor='black',
        capsize=5
    )

# Step 5: Add annotations and brackets
for basin, percentage_increase in grouped_changes.items():
    indices = [
        i for i, basin_ in enumerate(updated_df['BASIN_NAME'])
        if basin in basin_
    ]

    # Determine start and end of group
    start_idx = bar_positions[min(indices)]
    end_idx = bar_positions[max(indices)]
    max_y = after_data[after_data['BASIN_NAME'] == basin]['Upper_bound'].max() + 0.5  # Adjusted maximum y-axis value for annotations

    # Add bracket
    bracket_y_start = max_y - 0.2
    bracket_y_end = max_y
    ax.vlines(x=start_idx - 0.7, ymin=bracket_y_start, ymax=bracket_y_end, color='black', linewidth=1.5)
    ax.vlines(x=end_idx + 0.7, ymin=bracket_y_start, ymax=bracket_y_end, color='black', linewidth=1.5)
    ax.plot([start_idx - 0.7, end_idx + 0.7], [max_y, max_y], color='black', linewidth=1.5)
    basin_name = 'Fort Worth' if 'Fort Worth' in basin else basin.replace('Basin', '').strip()
    basin_name = 'Appalachian' if 'Appalachian' in basin_name else basin_name

    # Add annotation with basin name and percentage
#     ax.text(
#         (start_idx + end_idx) / 2,
#         max_y + 1,
#         f'{basin_name}\n+{percentage_increase:.0f}%'.replace(' (combined)', ''),
#         ha='center',
#         va='center',
#         fontsize=fontsize - 2,
#         bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white')
#     )
    
    
    
        # Determine bounds for error propagation
    before = before_data[before_data['BASIN_NAME'] == basin]
    after = after_data[after_data['BASIN_NAME'] == basin]

    mean_before = before['Fractional_loss'].mean()
    mean_after = after['Fractional_loss'].mean()
    err_before = (before['Upper_bound'] - before['Lower_bound']).mean() / 2
    err_after = (after['Upper_bound'] - after['Lower_bound']).mean() / 2

    # Compute percentage increase and its propagated uncertainty
    percent_change = (mean_after - mean_before) / mean_before * 100
    rel_uncertainty = np.sqrt((err_after / mean_after) ** 2 + (err_before / mean_before) ** 2)
    percent_error = rel_uncertainty * abs(percent_change)

    # Round for cleaner display
    percent_str = f"+{percent_change:.0f}%"
    error_bounds_str = f"[{(percent_change - percent_error):.0f}%, {(percent_change + percent_error):.0f}%]"
    percent_str_full = f"+{percent_change:.0f}[{(percent_change - percent_error):.0f}, {(percent_change + percent_error):.0f}]%"
    
    
    # Add annotation with basin name and bounds
    # Add annotation with basin name, % increase, and smaller error bounds
    # Draw main annotation (basin name + % increase)
    # Prepare text areas with different font sizes
    if basin_name == 'Appalachian (EOA)':
        basin_name = 'Appalachian'
    main_text = TextArea(f"{basin_name}\n+{percent_change:.0f}%", textprops=dict(fontsize=fontsize - 2, ha='center'))
    error_text = TextArea(f"[+{percent_change - percent_error:.0f}%, +{percent_change + percent_error:.0f}%]", 
                          textprops=dict(fontsize=fontsize - 12, color='black', ha='center'))

    # Stack vertically: main text on top, error bounds below
    packed_box = VPacker(children=[main_text, error_text], align="center", pad=0, sep=2)

    # Define location and place it
    ab = AnnotationBbox(
        packed_box,
        ((start_idx + end_idx) / 2, max_y + 1.5),
        box_alignment=(0.5, 0.5),
        frameon=True,
        bboxprops=dict(boxstyle="round,pad=1", edgecolor='black', facecolor='white')
    )
    ax.add_artist(ab)
    
max_y = after_data['Upper_bound'].max() + 3  # Adjusted maximum y-axis value for annotations
max_x = max(bar_positions) + 3
# Customize appearance
ax.set_xticks(bar_positions)

ax.tick_params(axis='x', labelsize=fontsize)  # Adjust the size of x-tick labels
ax.tick_params(axis='y', labelsize=fontsize)  # Adjust the size of y-tick labels
def percent_formatter(x, pos):
    return f"{x:.0f}%"
ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

ax.set_xticklabels(updated_df['Scenario'], rotation=45, ha='right', fontsize=fontsize)


major_ticks = np.linspace(0, 16, num=9)
ax.set_yticks(major_ticks)
minor_locator = MultipleLocator((major_ticks[1] - major_ticks[0]) / 2)  # Half the spacing of major ticks
ax.yaxis.set_minor_locator(MultipleLocator((major_ticks[1] - major_ticks[0]) / 2))
ax.set_yticks(np.arange(major_ticks[0], major_ticks[-1], (major_ticks[1] - major_ticks[0]) / 2), minor=True)  # Minor ticks only within major tick range

ax.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)  # Major grid lines
ax.grid(axis='y', which='minor', linestyle=':', linewidth=0.5, alpha=0.5)  # Minor grid lines (optional)

plt.ylim(0, max_y + 0.5)
plt.xlim(-2, max_x)
# plt.ylabel('Production-Normalized Methane Loss (%)', fontsize=fontsize)
plt.xlabel(None)
plt.legend(
    title='C1 molar fraction',
    fontsize=fontsize,
    title_fontsize=fontsize,
    loc='upper right',
    bbox_to_anchor=(1, 1),
    frameon=True,
    facecolor='white',
    edgecolor='black'
)
plt.tight_layout()

# Save and show the plot
plt.savefig('figures_out/fractional_loss_comparison.eps', format='eps', dpi=300)
plt.savefig('figures_out/fractional_loss_comparison.png', format='png', dpi=300)

plt.show()












