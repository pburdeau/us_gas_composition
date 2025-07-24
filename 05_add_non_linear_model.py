import warnings

warnings.filterwarnings('ignore')
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import math
from tqdm import tqdm
import glob
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.spatial import cKDTree

root_path = '/scratch/users/pburdeau/data/gas_composition'
out_path = root_path + '/out'

components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR',
              'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']


class MyDataset(Dataset):
    def __init__(self, df, alpha, i_min, i_max, comp):
        super(MyDataset, self).__init__()
        self.df = df
        # Store raw data directly
        self.gas = np.array(self.df['Monthly Gas'][i_min:i_max])
        self.oil = np.array(self.df['Monthly Oil'][i_min:i_max])
        #         self.gor = np.log(self.gas / (self.oil + 1e-3) + 1e-3) - 10. / 10.
        self.c1 = np.array(self.df[comp][i_min:i_max])

    def __len__(self):
        return len(self.gas)

    def __getitem__(self, idx):
        in_dict = {
            'input': torch.tensor([self.gas[idx], self.oil[idx]]).float(),
            'c1': torch.tensor([self.c1[idx]]).float(),
        }
        return in_dict


def load_model(model_path, n_layers, n_neurons):
    model = MyModel(n_layers, n_neurons)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to inference mode
    return model


def predict(model, df):
    # Prepare the data in the same way as the training dataset
    if 'Gas' in df.columns and 'Oil' in df.columns:
        selected_columns = ['Gas', 'Oil']
    elif 'Monthly Gas' in df.columns and 'Monthly Oil' in df.columns:
        selected_columns = ['Monthly Gas', 'Monthly Oil']
    else:
        raise ValueError(
            "Neither 'Gas' and 'Oil' nor 'Monthly Gas' and 'Monthly Oil' columns are available in the DataFrame.")
    inputs = torch.tensor(df[selected_columns].to_numpy(), dtype=torch.float32)  # Convert directly to tensor

    # Make predictions
    with torch.no_grad():
        c1_normalized, c1 = model(inputs)
    return c1.cpu().numpy()  # Return as a numpy array if more convenient


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


class MyModel(nn.Module):
    def __init__(self, n_layers, n_neurons, device, dropout_rate=0.2):
        super(MyModel, self).__init__()
        self.device = device  # Store device
        self.mlp = []
        in_dim = 3
        out_dim = n_neurons
        for i in range(n_layers):
            if i == n_layers - 1:
                out_dim = 1
            self.mlp.append(nn.Linear(in_dim, out_dim))
            self.mlp.append(nn.Dropout(dropout_rate))

            in_dim = out_dim
            if i < n_layers - 1:
                self.mlp.append(nn.ReLU())
            else:
                self.mlp.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*self.mlp).to(self.device)  # Ensure model is on the correct device

    def normalize(self, c1):
        normalized = (c1 / 100) ** 5.
        return torch.clamp(normalized, min=1e-6)

    def denormalize(self, c1_normalized):
        denormalized = (c1_normalized ** (1. / 5.)) * 100.
        return torch.clamp(denormalized, min=0., max=100.)

    def forward(self, x):
        x = x.to(self.device)  # Move input tensor to model's device

        gas, oil = x[:, 0], x[:, 1]

        gas_clipped = torch.clamp(gas, min=1e-3)
        oil_clipped = torch.clamp(oil, min=1e-3)
        gor_clipped = torch.clamp(gas_clipped / oil_clipped, min=1e-3, max=1e3)
        gor = torch.log(gor_clipped)

        gas = (torch.log(gas_clipped) - 10.) / 5.
        oil = (torch.log(oil_clipped) - 5.) / 4.

        x = torch.cat([gas[..., None], oil[..., None], gor[..., None]], -1).to(
            self.device)  # Ensure it's on the same device

        c1_normalized = torch.clamp(self.mlp(x), min=1e-6, max=1 - 1e-6)
        c1 = self.denormalize(c1_normalized)
        return c1_normalized, c1


def find_model(data_source, threshold, radius_factor, a, b, df, df_test, comp, batch_size, n_layers, n_neurons, lr,
               n_epochs, L, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_train = df.copy()
    # Ensure the columns exist
    if all(col in df_train.columns for col in ['X_prod', 'Y_prod', 'T_prod']) and all(
            col in df_test.columns for col in ['X_prod', 'Y_prod', 'T_prod']):
        common_rows = df_train.merge(df_test, on=['X_prod', 'Y_prod', 'T_prod'], how='inner')

        if not common_rows.empty:
            print(f"Warning: {len(common_rows)} test samples are also in the training set!")
        else:
            print("No test samples found in the training set.")
    else:
        print("X, Y, or T columns not found in one of the datasets.")

    alpha = torch.tensor(5., device=device)  # Ensure it's on the same device

    for dataset in [df, df_test]:
        dataset.dropna(subset=['Monthly Oil', 'Monthly Gas', comp], inplace=True)
        dataset.reset_index(drop=True, inplace=True)
        dataset[comp] = dataset[comp].astype(float)
        dataset['Monthly Gas'] = dataset['Monthly Gas'].astype(float)
        dataset['Monthly Oil'] = dataset['Monthly Oil'].astype(float)

    n_train = len(df)

    dataset = MyDataset(df, alpha, 0, len(df), comp)

    dataset_test = MyDataset(df_test, alpha, 0, len(df_test), comp)

    n_test = len(dataset_test)
    batch_size_test = 128 * (n_test // 128) if n_test >= 128 else n_test
    #     batch_size = min(512, len(df) // 10)
    #     batch_size = batch_size_test

    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            num_workers=4,
                            pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=True)

    if L == 1:
        def loss_fn(x, y):
            return torch.abs(x - y).mean()
    else:
        def loss_fn(x, y):
            return ((x - y) ** 2).mean()

    def train(n_layers, n_neurons, lr, n_epochs):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = MyModel(n_layers, n_neurons, device).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        #         n_epochs = 2000
        log_every = 1

        epoch_log = []
        loss_log = []
        mae_log = []

        MAE_rd = np.abs(dataset.c1 - np.median(dataset.c1)).mean()
        print(f'MAE for a random guess: {MAE_rd:.4e}')
        min_mae = 100
        final_mae = None

        results = []

        patience = 3
        best_mae = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            model.train()
            losses = []

            for idx, data_dict in enumerate(dataloader):
                optimizer.zero_grad()
                x = data_dict['input'].to(device)
                c1_normalized, c1 = model(x)
                loss = loss_fn(c1, data_dict['c1'].to(device))
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if epoch == 0 or (epoch + 1) % log_every == 0:
                MAE = []
                model.eval()
                with torch.no_grad():
                    for idx, data_dict in enumerate(dataloader_test):
                        x = data_dict['input'].to(device)
                        c1_normalized, c1 = model(x)
                        gt_c1 = data_dict['c1'].to(device)
                        MAE.append(torch.abs(c1 - gt_c1).mean().item())

                average_mae = np.mean(MAE)
                average_training_loss = np.mean(losses)
                epoch_log.append(epoch + 1)
                loss_log.append(average_training_loss)
                mae_log.append(average_mae)
                results.append([epoch + 1, average_training_loss, average_mae, n_layers, n_neurons, lr, batch_size])
                if average_mae < min_mae:
                    min_mae = average_mae
                final_mae = average_mae

                # === Early stopping check ===
                if average_mae < best_mae:
                    best_mae = average_mae
                    patience_counter = 0
                    best_model_state = model.state_dict()  # save best model
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1} — no improvement for {patience} epochs.")
                        break

                plt.figure(figsize=(10, 5))
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.plot(epoch_log, loss_log, label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss over Time')
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(epoch_log, mae_log, label='Test MAE', color='red')
                plt.xlabel('Epoch')
                plt.ylabel('Mean Absolute Error')
                plt.title('Test MAE over Time')
                plt.legend()

                plt.savefig(
                    f'/scratch/users/pburdeau/notebooks/parameters_sweep_{data_source}/parameters_swap_data_source_{data_source}_radius_factor_{radius_factor}_a_{a}_b_{b}_lr_{lr}_n_epochs_{n_epochs}_batch_size_{batch_size}_n_layers_{n_layers}_n_neurons_{n_neurons}_lr_{lr}_L_{L}.pdf')
        print(f'Final MAE: {final_mae:.4e}')
        return model, final_mae, min_mae

    model, final_mae, min_mae = train(n_layers, n_neurons, lr, n_epochs)
    return model, final_mae, min_mae


def filter_threshold_production(data_source, wells_data, other_data, REL_DIFF_THRESHOLD):
    if data_source == 'ghgrp':
        wells_data = wells_data[wells_data.Year >= 2015]

    gas_by_basin_1 = wells_data.groupby('BASIN_NAME')['Monthly Gas'].mean()
    gas_by_basin_2 = other_data.groupby('BASIN_NAME')['Monthly Gas'].mean()

    oil_by_basin_1 = wells_data.groupby('BASIN_NAME')['Monthly Oil'].mean()
    oil_by_basin_2 = other_data.groupby('BASIN_NAME')['Monthly Oil'].mean()

    gas_comparison = pd.DataFrame({
        'gas_1': gas_by_basin_1,
        'gas_2': gas_by_basin_2
    }).dropna()

    oil_comparison = pd.DataFrame({
        'oil_1': oil_by_basin_1,
        'oil_2': oil_by_basin_2
    }).dropna()

    gas_comparison['rel_diff_gas'] = abs(gas_comparison['gas_1'] - gas_comparison['gas_2']) / gas_comparison['gas_1']
    oil_comparison['rel_diff_oil'] = abs(oil_comparison['oil_1'] - oil_comparison['oil_2']) / oil_comparison['oil_1']
    valid_gas_basins = gas_comparison[gas_comparison['rel_diff_gas'] <= REL_DIFF_THRESHOLD].index
    valid_oil_basins = oil_comparison[oil_comparison['rel_diff_oil'] <= REL_DIFF_THRESHOLD].index
    valid_basins = valid_gas_basins.intersection(valid_oil_basins)
    filtered_other_data = other_data[other_data['BASIN_NAME'].isin(valid_basins)]
    return filtered_other_data


def apply_model_comp(filter_prod,
                     data_source,
                     a,
                     b,
                     df_data,
                     df_test,
                     df_validation,
                     test_block,
                     validation_block,
                     result_validation,
                     comp,
                     n_layers,
                     n_neurons,
                     batch_size,
                     n_epochs,
                     L,
                     lr,
                     result_all_new,
                     result_all,
                     result_all_nn,
                     result_sample_2):
    model_filename1 = f'model_{comp}_validation.pth'  # PyTorch model
    model_filename2 = f'model_{comp}_test.pth'

    def save_model(model, filename):
        """Save the trained model to disk."""
        torch.save(model.state_dict(), filename)

    def load_model(filename, n_layers, n_neurons):
        """Load the model from disk if available."""
        if os.path.exists(filename):
            model = MyModel(n_layers, n_neurons)  # Initialize model architecture
            model.load_state_dict(torch.load(filename))  # Load trained weights
            model.eval()  # Set to evaluation mode
            return model
        return None

    #     model1 = load_model(model_filename1, n_layers, n_neurons)
    #     model2 = load_model(model_filename2, n_layers, n_neurons)

    #     if model1 is None:
    concatenated_block = pd.concat([validation_block, test_block])

    df_train = df_data.merge(concatenated_block[['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME']],
                             on=['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME'],
                             how='left',
                             indicator=True)
    df_train = df_train[df_train['_merge'] == 'left_only'].drop(columns=['_merge'])
    if data_source == 'usgs':
        df_train = filter_non_hydrocarbons(df_train, threshold)
    model1, _, _ = find_model(data_source, threshold, radius_factor, a, b, df_train[df_train['Monthly Gas'] > 0],
                              df_test[df_test['Monthly Gas'] > 0],
                              comp,
                              batch_size,
                              n_layers,
                              n_neurons,
                              lr,
                              n_epochs,
                              L,
                              seed=42)
    #         save_model(model1, model_filename1)  # Save trained model

    merged_validation = pd.merge(result_validation,
                                 df_validation,  # contains only the original values from usgs_prod
                                 on=['X_prod',
                                     'Y_prod',
                                     'T_prod',
                                     'BASIN_NAME',
                                     'Monthly Oil',
                                     'Monthly Gas'],
                                 suffixes=('_predic_k', '_true'))

    merged_validation = merged_validation[~pd.isna(merged_validation[f'{comp}_true'])].reset_index(drop=True)
    merged_validation = merged_validation[~pd.isna(merged_validation[f'{comp}_predic_k'])].reset_index(drop=True)

    if filter_prod:
        wells_data = []
        basin_list = merged_validation.BASIN_NAME.unique()
        with tqdm(total=len(basin_list)) as pbar:
            for basin_name in basin_list:
                basin_file_path = os.path.join(root_path, f'wells_info_prod_per_basin/{basin_name}_final.csv')
                if os.path.isfile(basin_file_path):
                    basin_df = pd.read_csv(basin_file_path)
                    wells_data.append(basin_df)
                pbar.update(1)
        wells_data = pd.concat(wells_data, ignore_index=True)

        merged_validation = filter_threshold_production(data_source, wells_data, merged_validation, 1)


    predictions = predict(model1, merged_validation)
    merged_validation[f'{comp}_predic_gor'] = predictions

    # Find beta
    initial_beta = 0
    result = minimize(mae_weighted_variance, initial_beta,
                      args=(merged_validation[f'{comp}_true'],
                            merged_validation[f'{comp}_predic_k'],
                            merged_validation[f'std_{comp}'],
                            merged_validation[f'{comp}_predic_gor']),
                      bounds=[(0, 100)])

    optimal_beta = result.x[0]

    # removing only test block

    #     if model2 is None:
    concatenated_block = test_block

    df_train = df_data.merge(concatenated_block[['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME']],
                             on=['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME'],
                             how='left',
                             indicator=True)
    df_train = df_train[df_train['_merge'] == 'left_only'].drop(columns=['_merge'])

    if data_source == 'usgs':
        df_train = filter_non_hydrocarbons(df_train, threshold)

    model2, _, _ = find_model(data_source, threshold, radius_factor, a, b, df_train[df_train['Monthly Gas'] > 0],
                              df_test[df_test['Monthly Gas'] > 0],
                              comp,
                              batch_size,
                              n_layers,
                              n_neurons,
                              lr,
                              n_epochs,
                              L,
                              seed=42)
    #         save_model(model2, model_filename2)  # Save trained model

    # print the improvement

    merged_result = pd.merge(result_all,
                             test_block,
                             on=['X_prod',
                                 'Y_prod',
                                 'T_prod',
                                 'BASIN_NAME',
                                 'Monthly Oil',
                                 'Monthly Gas'],
                             suffixes=('_predic_k', '_true'))

    merged_result = merged_result[~pd.isna(merged_result[comp + '_true'])].reset_index(drop=True)
    merged_result = merged_result[~pd.isna(merged_result[comp + '_predic_k'])].reset_index(drop=True)

    if filter_prod:
        merged_result = filter_threshold_production(data_source, wells_data, merged_result, 0.9)

    predictions = predict(model2, merged_result)
    merged_result[f'{comp}_predic_gor'] = predictions

    merged_result = merged_result[~pd.isna(merged_result[f'{comp}_predic_gor'])]
    merged_result[comp] = predic_beta(merged_result[f'{comp}_predic_k'], merged_result[f'std_{comp}'],
                                      merged_result[f'{comp}_predic_gor'], optimal_beta)

    print('New mae: ', np.mean(np.abs(merged_result[comp] - merged_result[f'{comp}_true'])))
    print('Kriging mae: ', np.mean(np.abs(merged_result[f'{comp}_predic_k'] - merged_result[f'{comp}_true'])))
    print('Non-linear model mae: ',
          np.mean(np.abs(merged_result[f'{comp}_predic_gor'] - merged_result[f'{comp}_true'])))

    # compute the new values everywhere
    predictions = predict(model2, result_all_new)
    result_all_new[f'{comp}_predic_gor'] = predictions

    result_all_new = result_all_new[~pd.isna(result_all_new[f'{comp}_predic_gor'])]
    result_all_new[comp] = predic_beta(result_all_new[comp], result_all_new[f'std_{comp}'],
                                       result_all_new[f'{comp}_predic_gor'],
                                       optimal_beta)

    predictions = predict(model2, result_sample_2)
    result_sample_2[f'{comp}_predic_gor'] = predictions
    result_sample_2 = result_sample_2[~pd.isna(result_sample_2[f'{comp}_predic_gor'])]
    result_sample_2[f'{comp}_predic_kriging'] = result_sample_2[comp]
    result_sample_2[comp] = predic_beta(result_sample_2[comp], result_sample_2[f'std_{comp}'],
                                        result_sample_2[f'{comp}_predic_gor'], optimal_beta)

    predictions = predict(model2, result_all_new)
    result_all[f'{comp}_predic_gor'] = predictions

    predictions = predict(model2, result_all_nn)
    result_all_nn[f'{comp}_predic_gor'] = predictions

    result_all_new[f'beta_{comp}'] = optimal_beta
    result_all[f'beta_{comp}'] = optimal_beta
    result_all_nn[f'beta_{comp}'] = optimal_beta

    result_sample_2[f'beta_{comp}'] = optimal_beta

    return result_all_new, result_all, result_all_nn,  result_sample_2

def upload_result(all_prod_indices_all, path):
    csv_files_pattern = os.path.join(root_path, 'out', path)
    csv_files = glob.glob(csv_files_pattern)
    result_all_list = [pd.read_csv(file) for file in csv_files]
    result_all = pd.concat(result_all_list)

    result_all['X'] = (result_all['X'] * 6.34).round(1)
    result_all['Y'] = (result_all['Y'] * 6.34).round(1)

    # Prepare to collect merged results
    merged_list = []
    basins = result_all.BASIN_NAME.unique()

    with tqdm(total=len(basins)) as pbar:
        for basin in basins:
            basin_result = result_all[result_all.BASIN_NAME == basin].copy()
            basin_prod = all_prod_indices_all[all_prod_indices_all.BASIN_NAME == basin].copy()

            if basin_result.empty or basin_prod.empty:
                pbar.update(1)
                continue

            # Truncate production grid coords to match
            basin_prod['Prod_X_grid'] = basin_prod['Prod_X_grid'].round(1)
            basin_prod['Prod_Y_grid'] = basin_prod['Prod_Y_grid'].round(1)

            # Merge on rounded coordinates and time
            merged = pd.merge(
                basin_prod,
                basin_result,
                left_on=['Prod_X_grid', 'Prod_Y_grid', 'Prod_T_grid'],
                right_on=['X', 'Y', 'T'],
                suffixes=('', '_result')
            )

            # Drop redundant columns if needed
            merged = merged.drop(columns=['X_result', 'Y_result', 'T_result', 'BASIN_NAME_result'], errors='ignore')
            merged = merged.rename(columns={'X': 'X_prod', 'Y': 'Y_prod', 'T': 'T_prod'})

            merged_list.append(merged)
            pbar.update(1)

    final_merged = pd.concat(merged_list, ignore_index=True)
    return final_merged


def filter_non_hydrocarbons(df, threshold):
    # threshold can be 100
    # Define non-hydrocarbon columns
    non_hydrocarbons = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']
    df['non_hydrocarbon_sum'] = df[non_hydrocarbons].fillna(0).sum(axis=1)
    filtered_df = df[df['non_hydrocarbon_sum'] < threshold]

    filtered_df = filtered_df.drop(columns=['non_hydrocarbon_sum'])
    return filtered_df


def mae_weighted_variance(beta, true, pred1, pred_1_std, pred2):
    weight_krig = 1
    weight_pred = beta * (pred_1_std ** 2)
    sum_weights = weight_krig + weight_pred
    prediction = (pred1 + weight_pred * pred2) / sum_weights
    return np.mean(np.abs(true - prediction))

def predic_beta(pred1, pred_1_std, pred2, beta):
    weight_krig = 1
    weight_pred = beta * (pred_1_std ** 2)
    sum_weights = weight_krig + weight_pred
    prediction = (pred1 + weight_pred * pred2) / sum_weights
    return prediction

def return_merged(test_block, df_test, result_all_new, result_all, result_all_nn, comp):
    def prepare_and_analyze(df1, df2, comp):
        # Ensure only unique data from df2 that is not in df1
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        print('Merging ...', flush=True)
        df2_unique = df2_copy.merge(df1_copy, on=['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME'], how='left',
                                    indicator=True)
        df2_unique = df2_unique[df2_unique['_merge'] == 'left_only'].drop(columns=['_merge'])

        # Filter by BASIN_NAME before building the KDTree
        merged_data = []

        # Wrap the iteration over basins with tqdm for progress tracking
        for basin in tqdm(df1_copy['BASIN_NAME'].unique(), desc="Processing Basins"):
            df1_basin = df1_copy[df1_copy['BASIN_NAME'] == basin]
            df2_basin_unique = df2_unique[df2_unique['BASIN_NAME'] == basin]

            if not df2_basin_unique.empty:
                # Build a KDTree for efficient distance computations
                tree = KDTree(df2_basin_unique[['X_prod', 'Y_prod', 'T_prod']])
                closest_distances = tree.query(df1_basin[['X_prod', 'Y_prod', 'T_prod']].to_numpy())[0]

                # Assign distances back to df1_copy
                df1_copy.loc[df1_basin.index, 'closest_distance'] = closest_distances

        # Compare to model
        merged_basin_data = pd.merge(df1_copy, df2_copy, on=['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME'])
        merged_basin_data = merged_basin_data[~pd.isna(merged_basin_data[comp + '_x'])]
        merged_basin_data = merged_basin_data[~pd.isna(merged_basin_data[comp + '_y'])]

        # Calculate absolute error
        merged_basin_data['abs_error'] = np.abs(merged_basin_data[comp + '_y'] - merged_basin_data[comp + '_x'])
        merged_basin_data['sq_error'] = (merged_basin_data[comp + '_y'] - merged_basin_data[comp + '_x']) ** 2
        merged_basin_data['nse'] = ((merged_basin_data[comp + '_y'] - merged_basin_data[comp + '_x']) /
                                    merged_basin_data['std_C1']) ** 2
        print('Merged.', flush=True)
        return merged_basin_data

    # Merge with an indicator column
    merged = df_test.merge(test_block[['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME']],
                           on=['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME'],
                           how='right',
                           indicator=True)

    # Set NA for specific columns where the row is not in df1
    cols_to_null = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR',
                    'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']  # Replace with actual column names
    merged.loc[merged['_merge'] == 'left_only', cols_to_null] = pd.NA

    # Drop the indicator column
    merged.drop(columns=['_merge'], inplace=True)

    # Resulting DataFrame
    df1 = merged.reset_index(drop=True)

    df2_kriging = result_all_new.reset_index(drop=True)  # Define your actual df2 for kriging

    df2_simple_kriging = result_all[~pd.isna(result_all[comp + '_predic_gor'])].reset_index(
        drop=True)  # Define your actual df2 for simple kriging

    df2_nn = result_all_nn[~pd.isna(result_all_nn[comp + '_predic_gor'])].reset_index(
        drop=True)


    merged_kriging = prepare_and_analyze(df1.copy(), df2_kriging, comp)

    merged_simple_kriging = prepare_and_analyze(df1.copy(), df2_simple_kriging, comp)
    merged_nn = prepare_and_analyze(df1.copy(), df2_nn, comp)

    return merged_kriging, merged_simple_kriging, merged_nn

def export_results(threshold, data_source, keep_nanapis, radius_factor, filter_prod):
    if data_source == 'ghgrp':

        all_prod_indices_all = pd.read_csv(f'/scratch/users/pburdeau/notebooks/all_prod_indices_all_ghgrp.csv')
    else:
        all_prod_indices_all = pd.read_csv(f'/scratch/users/pburdeau/notebooks/all_prod_indices_all_{threshold}.csv')
    all_prod_indices_all['X'] = (all_prod_indices_all['X']).round(1)
    all_prod_indices_all['Y'] = (all_prod_indices_all['Y']).round(1)

    all_prod_indices_all['Prod_X_grid'] = all_prod_indices_all['Prod_X_grid'].round(1)
    all_prod_indices_all['Prod_Y_grid'] = all_prod_indices_all['Prod_Y_grid'].round(1)

    with tqdm(total=len([-1, 1, -1, 1])) as pbar:
        for a, b in zip([-1, 1, -1, 1], [-1, 1, 1, -1]):
            a = str(a)
            b = str(b)
            result_all = upload_result(all_prod_indices_all,
                                       f'result_data_source_{data_source}_a_{a}_b_{b}_basin_*_sample_1_interp_ordinary_kriging_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')
            result_sample_2 = upload_result(all_prod_indices_all,
                                            f'result_data_source_{data_source}_a_{a}_b_{b}_basin_*_sample_2_interp_ordinary_kriging_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')
            result_all_nn = upload_result(all_prod_indices_all,
                                          f'result_data_source_{data_source}_a_{a}_b_{b}_basin_*_sample_1_interp_nn_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')

            result_validation = upload_result(all_prod_indices_all,
                                              f'result_data_source_{data_source}_a_{a}_b_{b}_basin_*_sample_0_interp_ordinary_kriging_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')
            validation_block = upload_result(all_prod_indices_all,
                                             f'block_validation_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_sample_0_a_{a}_b_{b}_basin*_radius_factor_{radius_factor}.csv')
            test_block = upload_result(all_prod_indices_all,
                                       f'block_test_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_sample_0_a_{a}_b_{b}_basin_*_radius_factor_{radius_factor}.csv')

            if data_source == 'usgs':
                df_data = pd.read_csv(os.path.join(root_path, 'usgs/usgs_prod.csv'))
            else:
                df_data = pd.read_csv(os.path.join(root_path, 'ghgrp/ghgrp_production_processed.csv'))

                df_data['X_prod'] = df_data['X']
                df_data['Y_prod'] = df_data['Y']
                df_data['T_prod'] = df_data['T']

            df_data['X_prod'] = (df_data['X_prod']).round(1)
            df_data['Y_prod'] = (df_data['Y_prod']).round(1)

            if data_source == 'usgs':
                df_data = filter_non_hydrocarbons(df_data, threshold)


            df_test = test_block[['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME']].merge(df_data,
                                                                                     on=['X_prod', 'Y_prod', 'T_prod',
                                                                                         'BASIN_NAME'])

            if data_source == 'usgs':
                df_test = filter_non_hydrocarbons(df_test, threshold)

            df_validation = validation_block[['X_prod', 'Y_prod', 'T_prod', 'BASIN_NAME']].merge(df_data,
                                                                                                 on=['X_prod', 'Y_prod',
                                                                                                     'T_prod',
                                                                                                     'BASIN_NAME'])
            if data_source == 'usgs':
                df_validation = filter_non_hydrocarbons(df_validation, threshold)

            result_all_new = result_all.copy()  # will progressively contain the new values from neural network

            comp = 'C1'

            if data_source == 'usgs':
                n_layers = 3
                n_neurons = 16
                batch_size = 128
                n_epochs = 500
                L = 2
                lr = 1e-4
            else:
                n_layers = 3
                n_neurons = 16
                batch_size = 1024
                n_epochs = 30
                L = 2
                lr = 1e-3
            print(f'Applying model for {comp}', flush=True)

            result_all_new, result_all, result_all_nn, result_sample_2 = apply_model_comp(
                filter_prod,
                data_source,
                a,
                b,
                df_data,
                df_test,
                df_validation,
                test_block,
                validation_block,
                result_validation,
                comp,
                n_layers,
                n_neurons,
                batch_size,
                n_epochs,
                L,
                lr,
                result_all_new,
                result_all,
                result_all_nn,
                result_sample_2)

            if data_source == 'usgs':
                comp = 'C2'
                print(f'Applying model for {comp}', flush=True)

                n_layers = 3
                n_neurons = 16
                batch_size = 256
                n_epochs = 2500
                L = 2
                lr = 3e-5
                result_all_new, result_all, result_all_nn, result_sample_2 = apply_model_comp(
                    filter_prod,
                    data_source,
                    a,
                    b,
                    df_data,
                    df_test,
                    df_validation,
                    test_block,
                    validation_block,
                    result_validation,
                    comp,
                    n_layers,
                    n_neurons,
                    batch_size,
                    n_epochs,
                    L,
                    lr,
                    result_all_new,
                    result_all,
                    result_all_nn,
                    result_sample_2)
            print(f'Saving results...', flush=True)

            result_all_new.to_csv(
                out_path + f'/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_all_new_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv')
            result_all.to_csv(
                out_path + f'/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_all_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv')
            result_sample_2.to_csv(
                out_path + f'/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_sample_2_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv')
            result_all_nn.to_csv(
                out_path + f'/data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_all_nn_a_{a}_b_{b}_radius_factor_{radius_factor}_filter_{filter_prod}.csv')

            print(f'Finished a = {a}, b = {b}', flush=True)
            pbar.update(1)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=int)
    parser.add_argument('--threshold', type=int)
    parser.add_argument('--radius_factor', type=int)
    parser.add_argument('--filter_prod', type=int)

    arguments = parser.parse_args()

    data_sources = ['usgs', 'ghgrp']
    data_source = data_sources[arguments.data_source]

    thresholds = [1000, 10, 25, 50]
    threshold = thresholds[arguments.threshold]

    radius_factors = [0.175, 0.15]
    radius_factor = radius_factors[arguments.radius_factor]
    keep_nanapis = True

    filter_prods = [True, False]
    filter_prod = filter_prods[arguments.filter_prod]

    export_results(threshold, data_source, keep_nanapis, radius_factor, filter_prod)