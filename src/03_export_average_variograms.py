import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import skgstat as skg
from skgstat import models
import argparse
import json

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

    for zz in list_zz:
        list_cols = ['X', 'Y', 'T', zz]

        np_data = df[list_cols].to_numpy().astype(float)
        origin = np.array([xmin, ymin, tmin])
        resolution = np.array([res_space, res_space, res_time])

        grid_indices = np.rint(((np_data[:, :3] - origin) / resolution)).astype(int)
        grid_shape = (len(x_range), len(y_range), len(t_range))
        grid_sum = np.zeros(grid_shape)
        grid_count = np.zeros(grid_shape)

        for i in range(np_data.shape[0]):
            x_idx, y_idx, t_idx = grid_indices[i, :3]
            if x_idx < grid_shape[0] and y_idx < grid_shape[1] and t_idx < grid_shape[2]:
                grid_sum[x_idx, y_idx, t_idx] += np_data[i, 3]
                grid_count[x_idx, y_idx, t_idx] += 1

        if mean_agg:
            with np.errstate(invalid='ignore', divide='ignore'):
                grid_matrix = np.divide(grid_sum, grid_count)
        else:
            grid_matrix = grid_sum

        flat_grid_matrix = grid_matrix.ravel()
        df_grid[zz] = flat_grid_matrix
    return df_grid


def create_variogram(coords, values, n_lags, maxlag):
    V1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, maxlag=maxlag, normalize=False)
    azimuth = 0
    nugget = V1.parameters[2]
    major_range = V1.parameters[0]
    minor_range = V1.parameters[0]
    sill = V1.parameters[1]
    vtype = 'Exponential'

    return azimuth, nugget, major_range, minor_range, sill, vtype


def plot_variogram(V1, title, maxlag, output_plot):
    xdata = V1.bins
    ydata = V1.experimental
    sill = V1.parameters[1]
    range_val = V1.parameters[0]

    V1.model = 'exponential'
    xi = np.linspace(0, xdata[-1], 100)
    y_exp = [models.exponential(h, V1.parameters[0], V1.parameters[1], V1.parameters[2]) for h in xi]

    # plot variogram model
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(xdata * 6.34 / 1000, ydata, 'og', label="Experimental variogram", color='#adc2da')
    ax.plot(xi * 6.34 / 1000, y_exp, 'b-', label='Exponential variogram', color='#3a4659')

    # Add annotations for sill and range
    text_x = max(xi) * 6.34 / 1000 * 0.7
    text_y = max(y_exp) * 0.8

    ax.text(text_x, text_y, f"Sill: {sill:.1f} \nRange: {range_val * 6.34 / 1000:.1f} km",
            fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Lag [km]', fontsize=14)
    ax.set_ylabel(r'Semivariance', fontsize=14)

    # Adjust legend font size
    ax.legend(loc='lower right', fontsize=14)

    plt.savefig(output_plot)
    plt.show()



def average_variogram(basins_for_average_variogram, data_source, df_prod, res_space, res_time, alpha, maxlag, n_lags,
                      keep_nanapis, threshold):
    if data_source == 'usgs':
        if keep_nanapis:
            data = pd.read_csv(os.path.join(root_path, 'usgs/usgs_processed_with_nanapis.csv'))
        else:
            data = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_without_nanapis.csv'))

        def filter_non_hydrocarbons(df, threshold):
            non_hydrocarbons = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']
            df['non_hydrocarbon_sum'] = df[non_hydrocarbons].fillna(0).sum(axis=1)
            return df[df['non_hydrocarbon_sum'] < threshold].drop(columns=['non_hydrocarbon_sum'])

        data = filter_non_hydrocarbons(data, threshold)
    else:
        data = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))

    # Directory for saving results
    output_dir = os.path.join(root_path, 'average_variograms', f'{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_1year')
    os.makedirs(output_dir, exist_ok=True)

    # Filter data for specified basins
    df_data = data.reset_index(drop=True)
    df_data = df_data[df_data.BASIN_NAME.isin(basins_for_average_variogram)].reset_index(drop=True)
    df_data['X'] = df_data['X'] / alpha
    df_data['Y'] = df_data['Y'] / alpha

    # Components to process
    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
    components = [comp for comp in components if comp in df_data.columns]

    if not components:
        raise ValueError("No valid components found in df_data columns.")

    for comp in components:
        print(f"Processing component: {comp}")

        combined_coords = []
        combined_values = []

        for basin in basins_for_average_variogram:
            basin_data = df_data[df_data.BASIN_NAME == basin].reset_index(drop=True)
            grid, grid_params = create_grid(df_prod[df_prod.BASIN_NAME == basin], res_space, res_time)
            prod_on_grid = aggregate(df_prod[df_prod.BASIN_NAME == basin], ['Gas'], grid_params, mean_agg=False)

            data_on_grid = aggregate(basin_data, [comp], grid_params, mean_agg=True)
            data_on_grid = data_on_grid[prod_on_grid.Gas > 0].reset_index(drop=True)

            data_single = data_on_grid[~pd.isna(data_on_grid[comp])].reset_index(drop=True)
            data_single = data_single[data_single[comp] > 0].reset_index(drop=True)

            if not data_single.empty:
                # Normalize data
                transformer = QuantileTransformer(n_quantiles=500, output_distribution='normal')
                data_single['norm_' + comp] = transformer.fit_transform(
                    data_single[comp].values.reshape(-1, 1)).flatten()

                coords = data_single[['X', 'Y', 'T']].values
                values = data_single['norm_' + comp].values

                combined_coords.append(coords)
                combined_values.append(values)

        # Combine data across all basins for this component
        if combined_coords:
            all_coords = np.vstack(combined_coords)
            all_values = np.hstack(combined_values)

            # Compute variogram for the combined data
            azimuth, nugget, major_range, minor_range, sill, vtype = create_variogram(all_coords, all_values, n_lags, maxlag)
            variogram_data = {
                "azimuth": azimuth,
                "nugget": nugget,
                "major_range": major_range,
                "minor_range": minor_range,
                "sill": sill,
                "type": vtype
            }

            # Save variogram as JSON
            variogram_file = os.path.join(output_dir, f'{comp}_variogram.json')
            with open(variogram_file, 'w') as f:
                json.dump(variogram_data, f, indent=4)
            print(f"Saved variogram for {comp} to {variogram_file}")

            # Plot and save the variogram
            output_plot = os.path.join(output_dir, f'{comp}_variogram.pdf')
            plot_variogram(skg.Variogram(all_coords, all_values, bin_func='even', n_lags=n_lags, maxlag=maxlag),
                           f"Combined Variogram for {comp}", maxlag, output_plot)

    print(f"All variograms saved in {output_dir}")



def my_function(data_source, keep_nanapis, threshold):
    basins_for_average_variogram = ['Appalachian Basin',
                                    'Appalachian Basin (Eastern Overthrust Area)',
                                    'Permian Basin',
                                    'Arkla Basin',
                                    'Anadarko Basin',
                                    'Denver Basin',
                                    'Green River Basin',
                                    'Arkoma Basin',
                                    'Gulf Coast Basin (LA, TX)',
                                    'East Texas Basin']

    base_path = os.path.join(root_path, 'wells_info_prod_per_basin')

    combined_data = pd.DataFrame()

    for basin in basins_for_average_variogram:
        file_path = os.path.join(base_path, f'{basin}_final.csv')
        print(f"Reading data for basin: {basin}")
        basin_data = pd.read_csv(file_path)
        combined_data = pd.concat([combined_data, basin_data], ignore_index=True)

    alpha = 6.34
    maxlag = 100000 / alpha
    n_lags = 10
    k = 100  # number of neighboring data points used to estimate a given point
    rad = maxlag
    res_space = 2000 / alpha  # 1km
    res_time = 365  # 1 years

    df_prod = combined_data.reset_index(drop=True)
    df_prod = df_prod.rename(columns={'Monthly Oil': 'Oil', 'Monthly Gas': 'Gas'})
    df_prod = df_prod[df_prod.Gas > 0].reset_index(drop=True)
    df_prod['Oil'].fillna(0, inplace=True)
    df_prod['X'] = df_prod['X'] / alpha
    df_prod['Y'] = df_prod['Y'] / alpha
    df_prod = df_prod[~pd.isna(df_prod['Gas'])].reset_index(drop=True)

    average_variogram(basins_for_average_variogram, data_source, df_prod, res_space, res_time, alpha, maxlag, n_lags,
                      keep_nanapis, threshold)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=int)
    parser.add_argument('--threshold', type=int)
    parser.add_argument('--keep_nanapis', type=int)
    arguments = parser.parse_args()

    data_sources = ['usgs', 'ghgrp']
    data_source = data_sources[arguments.data_source]

    # thresholds = [5, 10, 15, 20, 25, 50, 1000]
    thresholds = [1000, 25, 50, 10]

    threshold = thresholds[arguments.threshold]

    keep_nanapiss = [True, False]
    keep_nanapis = keep_nanapiss[arguments.keep_nanapis]

    my_function(data_source, keep_nanapis, threshold)

