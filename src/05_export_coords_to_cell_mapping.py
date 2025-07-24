import os
import pandas as pd
from tqdm import tqdm
import numpy as np

root_path = '/scratch/users/pburdeau/data/gas_composition'

def create_grid(df, res_space, res_time):
    xmin, xmax = df['X'].min(), df['X'].max()
    ymin, ymax = df['Y'].min(), df['Y'].max()
    tmin, tmax = df['T'].min(), df['T'].max()
    grid_params = (xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time)

    # Create 3D grid coordinates
    x_range = np.arange(xmin, xmax + res_space, res_space)
    y_range = np.arange(ymin, ymax + res_space, res_space)
    t_range = np.arange(tmin, tmax + res_time, res_time)
    grid_coord = np.meshgrid(x_range, y_range, t_range, indexing='ij')
    grid = np.column_stack([g.ravel() for g in grid_coord])
    return grid, grid_params


def aggregate(df, list_zz, grid_params, mean_agg=True):
    # returns also initial coordinates of data

    xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time = grid_params
    x_range = np.arange(xmin, xmax + res_space, res_space)
    y_range = np.arange(ymin, ymax + res_space, res_space)
    t_range = np.arange(tmin, tmax + res_time, res_time)
    grid_coord = np.meshgrid(x_range, y_range, t_range, indexing='ij')

    df = df[(df.X >= xmin) & (df.X <= xmax)
            & (df.Y >= ymin) & (df.Y <= ymax)
            & (df['T'] >= tmin) & (df['T'] <= tmax)].reset_index(drop=True)
    # Flatten the grid for DataFrame representation
    flat_grid_coords = np.column_stack([g.ravel() for g in grid_coord])

    df_grid = pd.DataFrame(flat_grid_coords, columns=['X', 'Y', 'T'])
    indices_df = df[['X', 'Y', 'T']].copy()
    #     print(indices_df)

    for zz in list_zz:
        list_cols = ['X', 'Y', 'T']
        indices_df[zz] = df[zz]
        list_cols.append(zz)

        # Convert DataFrame to NumPy array for processing
        np_data = df[list_cols].to_numpy().astype(float)
        origin = np.array([xmin, ymin, tmin])
        resolution = np.array([res_space, res_space, res_time])

        # Calculate grid indices for each data point
        grid_indices = np.rint(((np_data[:, :3] - origin) / resolution)).astype(int)
        indices_df[f'{zz}_grid_index'] = list(zip(grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]))

        # Initialize 3D aggregation arrays
        grid_shape = (len(x_range), len(y_range), len(t_range))
        grid_sum = np.zeros(grid_shape)
        grid_count = np.zeros(grid_shape)
        # Aggregate data into the grid
        for i in range(np_data.shape[0]):
            x_idx, y_idx, t_idx = grid_indices[i, :3]
            if x_idx < grid_shape[0] and y_idx < grid_shape[1] and t_idx < grid_shape[2]:
                grid_sum[x_idx, y_idx, t_idx] += np_data[i, 3]
                grid_count[x_idx, y_idx, t_idx] += 1

        if mean_agg == True:
            # Calculate average values where count is not zero
            with np.errstate(invalid='ignore', divide='ignore'):
                grid_matrix = np.divide(grid_sum, grid_count)
        #             grid_matrix = np.nan_to_num(grid_matrix)  # Replace NaNs with 0
        else:
            grid_matrix = grid_sum

        flat_grid_matrix = grid_matrix.ravel()

        # update DataFrame
        df_grid[zz] = flat_grid_matrix
        # Add gridded coordinates to indices_df
        indices_df[f'{zz}_X_grid'] = x_range[grid_indices[:, 0]]
        indices_df[f'{zz}_Y_grid'] = y_range[grid_indices[:, 1]]
        indices_df[f'{zz}_T_grid'] = t_range[grid_indices[:, 2]]
    return df_grid, indices_df


def create_data_indices_all(data_source, keep_nanapis, threshold):
    if data_source == 'usgs':
        if keep_nanapis:
            data = pd.read_csv(os.path.join(root_path, 'usgs/usgs_processed_with_nanapis.csv'))  # all basins
        else:
            data = pd.read_csv(os.path.join(root_path, 'usgs/usgs_processed_without_nanapis.csv'))  # all basins

        def filter_non_hydrocarbons(df, threshold):
            # threshold can be 100
            # Define non-hydrocarbon columns
            non_hydrocarbons = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']
            df['non_hydrocarbon_sum'] = df[non_hydrocarbons].fillna(0).sum(axis=1)
            filtered_df = df[df['non_hydrocarbon_sum'] < threshold]

            filtered_df = filtered_df.drop(columns=['non_hydrocarbon_sum'])
            return filtered_df

        data = filter_non_hydrocarbons(data, threshold)

    else:
        data = pd.read_csv(os.path.join(root_path, 'ghgrp/ghgrp_production_processed.csv'))  # all basins

    alpha = 6.34
    res_space = 2000 / alpha
    res_time = 1825

    grid_params_dic = {}
    data_indices_list = []
    prod_indices_list = []
    prod_oil_indices_list = []

    basins_many_datapoints = ['Anadarko Basin',
                              'Gulf Coast Basin (LA, TX)',
                              'Permian Basin',
                              'Appalachian Basin',
                              'Appalachian Basin (Eastern Overthrust Area)',
                              'Chautauqua Platform',
                              'Arkoma Basin',
                              'San Juan Basin',
                              'Green River Basin',
                              'Paradox Basin',
                              'Piceance Basin',
                              'Palo Duro Basin',
                              'East Texas Basin',
                              'Denver Basin',
                              'Arkla Basin',
                              'South Oklahoma Folded Belt',
                              'Bend Arch',
                              'Fort Worth Syncline',
                              'Wind River Basin',
                              'Uinta Basin',
                              'Williston Basin',
                              'Las Animas Arch']
    basins_all = data.BASIN_NAME.unique()

    with tqdm(total=len(basins_all)) as pbar:
        for basin in basins_all:
            basin_file_path = os.path.join(root_path, f'wells_info_prod_per_basin/{basin}_final.csv')
            if os.path.exists(basin_file_path) and os.path.getsize(basin_file_path) > 0:
                df_data = data.reset_index(drop=True)
                df_data = df_data[df_data.BASIN_NAME == basin].reset_index(drop=True)
                df_data['X'] = df_data.X / alpha
                df_data['Y'] = df_data.Y / alpha

                wells_data = pd.read_csv(basin_file_path)  # only basin data

                df_prod = wells_data.reset_index(drop=True)
                df_prod = df_prod[df_prod['Monthly Gas'] > 0].reset_index(drop=True)
                if not df_prod.empty:
                    df_prod['X'] = df_prod.X / alpha
                    df_prod['Y'] = df_prod.Y / alpha
                    df_prod = df_prod[~pd.isna(df_prod['Monthly Gas'])].reset_index(drop=True)

                    grid, grid_params = create_grid(df_prod, res_space, res_time)
                    grid_params_dic[basin] = grid_params

                    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5',
                                  'I-C5', 'C6+']
                    components = [comp for comp in components if comp in df_data.columns]

                    data_on_grid, data_indices = aggregate(df_data, components, grid_params, mean_agg=True)
                    prod_on_grid, prod_indices = aggregate(df_prod, ['Monthly Gas'], grid_params, mean_agg=False)
                    prod_oil_on_grid, prod_oil_indices = aggregate(df_prod, ['Monthly Oil'], grid_params,
                                                                   mean_agg=False)
                    data_on_grid = data_on_grid[prod_on_grid['Monthly Gas'] > 0].reset_index(drop=True)
                    prod_oil_on_grid = prod_oil_on_grid[prod_on_grid['Monthly Gas'] > 0].reset_index(drop=True)
                    prod_on_grid = prod_on_grid[prod_on_grid['Monthly Gas'] > 0].reset_index(drop=True)

                    data_indices['BASIN_NAME'] = basin
                    prod_indices['BASIN_NAME'] = basin
                    prod_oil_indices['BASIN_NAME'] = basin

                    data_indices_list.append(data_indices)
                    prod_indices_list.append(prod_indices)
                    prod_oil_indices_list.append(prod_oil_indices)

            pbar.update(1)

    data_indices_all = pd.concat(data_indices_list, ignore_index=True)
    prod_indices_all = pd.concat(prod_indices_list, ignore_index=True)
    prod_oil_indices_all = pd.concat(prod_oil_indices_list, ignore_index=True)
    print('Rescaling coordinates...')
    data_indices_all['X'] = data_indices_all['X'] * alpha
    data_indices_all['Y'] = data_indices_all['Y'] * alpha

    prod_indices_all['X'] = prod_indices_all['X'] * alpha
    prod_indices_all['Y'] = prod_indices_all['Y'] * alpha

    prod_oil_indices_all['X'] = prod_oil_indices_all['X'] * alpha
    prod_oil_indices_all['Y'] = prod_oil_indices_all['Y'] * alpha

    for comp in components:
        data_indices_all[f'{comp}_X_grid'] *= alpha
        data_indices_all[f'{comp}_Y_grid'] *= alpha

    prod_indices_all[f'Monthly Gas_X_grid'] *= alpha
    prod_indices_all[f'Monthly Gas_Y_grid'] *= alpha

    prod_oil_indices_all[f'Monthly Oil_X_grid'] *= alpha
    prod_oil_indices_all[f'Monthly Oil_Y_grid'] *= alpha

    print('Renaming columns...')

    prod_indices_all = prod_indices_all.rename(columns={'Monthly Gas_grid_index': 'Prod_index',
                                                        'Monthly Gas_X_grid': 'Prod_X_grid',
                                                        'Monthly Gas_Y_grid': 'Prod_Y_grid',
                                                        'Monthly Gas_T_grid': 'Prod_T_grid'})

    prod_oil_indices_all = prod_oil_indices_all.rename(columns={'Monthly Oil_grid_index': 'Prod_index',
                                                                'Monthly Oil_X_grid': 'Prod_X_grid',
                                                                'Monthly Oil_Y_grid': 'Prod_Y_grid',
                                                                'Monthly Oil_T_grid': 'Prod_T_grid'})
    print('Grouping and summing oil ...')

    summed_duplicates_oil = prod_oil_indices_all.groupby(['X', 'Y', 'T', 'Prod_index', 'Prod_X_grid',
                                                          'Prod_Y_grid', 'Prod_T_grid', 'BASIN_NAME'], as_index=False)[
        ['Monthly Oil']].sum()
    print('Grouping and summing gas ...')

    summed_duplicates_gas = prod_indices_all.groupby(['X', 'Y', 'T', 'Prod_index', 'Prod_X_grid',
                                                      'Prod_Y_grid', 'Prod_T_grid', 'BASIN_NAME'], as_index=False)[
        ['Monthly Gas']].sum()
    print('Merging oil and gas ...')

    all_prod_indices_all = pd.merge(summed_duplicates_oil, summed_duplicates_gas,
                                    on=['X', 'Y', 'T', 'Prod_index', 'Prod_X_grid',
                                        'Prod_Y_grid', 'Prod_T_grid', 'BASIN_NAME'])

    return data_indices_all, all_prod_indices_all

if __name__ == "__main__":
    data_source = 'usgs'  # or 'ghgrp'
    keep_nanapis = True
    threshold = 1000

    data_indices_all, all_prod_indices_all = create_data_indices_all(data_source, keep_nanapis, threshold)