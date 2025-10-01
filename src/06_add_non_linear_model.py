#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid kriging + GOR-based ML refinement

Flow
----
1) Load gridded kriging results & production voxel indices (per basin).
2) Build train/val/test splits by spatial blocks (already created upstream).
3) Train a small MLP to predict component from (Gas, Oil, GOR) on training cells.
4) Calibrate a blending weight β by minimizing MAE on validation cells:
      y_hat = (y_krig + β * σ_krig^2 * y_mlp) / (1 + β * σ_krig^2)
5) Re-fit ML model removing only the *test* block, then apply β to:
      (a) test cells (for reporting), and
      (b) all cells (to produce refined surfaces).
6) Export refined CSVs for each (a,b) block split.

"""

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
from scipy.spatial import cKDTree  # (unused but kept as in your original)

# ------------------------------ #
#           CONFIG PATHS         #
# ------------------------------ #

root_path = '/scratch/users/pburdeau/data/gas_composition'
out_path = os.path.join(root_path, 'out')
os.makedirs(out_path, exist_ok=True)

# ------------------------------ #
#          GLOBAL LISTS          #
# ------------------------------ #

components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR',
              'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']

# ------------------------------ #
#            DATASETS            #
# ------------------------------ #

class MyDataset(Dataset):
    """
    Tiny dataset wrapper for (Gas, Oil) -> component target (comp)
    """
    def __init__(self, df, alpha, i_min, i_max, comp):
        super(MyDataset, self).__init__()
        self.df = df
        self.gas = np.array(self.df['Monthly Gas'][i_min:i_max])
        self.oil = np.array(self.df['Monthly Oil'][i_min:i_max])
        self.c1  = np.array(self.df[comp][i_min:i_max]).astype(float)

    def __len__(self):
        return len(self.gas)

    def __getitem__(self, idx):
        return {
            'input': torch.tensor([self.gas[idx], self.oil[idx]], dtype=torch.float32),
            'c1':    torch.tensor([self.c1[idx]], dtype=torch.float32),
        }

# ------------------------------ #
#              GRID              #
# ------------------------------ #

def aggregate(df, list_zz, grid_params, mean_agg=True):
    """
    Aggregate df values onto the (X,Y,T) grid and return:
      df_grid: flattened grid with aggregated values
      indices_df: original rows with mapped voxel indices and centers
    """
    xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time = grid_params
    x_range = np.arange(xmin, xmax + res_space, res_space)
    y_range = np.arange(ymin, ymax + res_space, res_space)
    t_range = np.arange(tmin, tmax + res_time, res_time)
    grid_coord = np.meshgrid(x_range, y_range, t_range, indexing='ij')

    df = df[(df.X >= xmin) & (df.X <= xmax)
            & (df.Y >= ymin) & (df.Y <= ymax)
            & (df['T'] >= tmin) & (df['T'] <= tmax)].reset_index(drop=True)

    flat_grid_coords = np.column_stack([g.ravel() for g in grid_coord])
    df_grid = pd.DataFrame(flat_grid_coords, columns=['X', 'Y', 'T'])
    indices_df = df[['X', 'Y', 'T']].copy()

    for zz in list_zz:
        list_cols = ['X', 'Y', 'T', zz]
        indices_df[zz] = df[zz]

        np_data = df[list_cols].to_numpy().astype(float)
        origin = np.array([xmin, ymin, tmin])
        resolution = np.array([res_space, res_space, res_time])

        grid_indices = np.rint(((np_data[:, :3] - origin) / resolution)).astype(int)
        indices_df[f'{zz}_grid_index'] = list(zip(grid_indices[:, 0], grid_indices[:, 1], grid_indices[:, 2]))

        shape = (len(x_range), len(y_range), len(t_range))
        grid_sum   = np.zeros(shape)
        grid_count = np.zeros(shape)

        for i in range(np_data.shape[0]):
            x_idx, y_idx, t_idx = grid_indices[i, :3]
            if 0 <= x_idx < shape[0] and 0 <= y_idx < shape[1] and 0 <= t_idx < shape[2]:
                grid_sum[x_idx, y_idx, t_idx] += np_data[i, 3]
                grid_count[x_idx, y_idx, t_idx] += 1

        if mean_agg:
            with np.errstate(invalid='ignore', divide='ignore'):
                grid_matrix = np.divide(grid_sum, grid_count)
        else:
            grid_matrix = grid_sum

        df_grid[zz] = grid_matrix.ravel()
        # map to voxel centers
        indices_df[f'{zz}_X_grid'] = x_range[np.clip(grid_indices[:, 0], 0, len(x_range)-1)]
        indices_df[f'{zz}_Y_grid'] = y_range[np.clip(grid_indices[:, 1], 0, len(y_range)-1)]
        indices_df[f'{zz}_T_grid'] = t_range[np.clip(grid_indices[:, 2], 0, len(t_range)-1)]

    return df_grid, indices_df


def create_grid(df, res_space, res_time):
    """
    Return (grid, grid_params) for inclusive bounds across X,Y,T.
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

# ------------------------------ #
#             MODEL              #
# ------------------------------ #

class MyModel(nn.Module):
    """
    Small MLP:
      input: [log(Gas), log(Oil), log(GOR)]
      output: normalized component in (0,1) then denormalized to %
    """
    def __init__(self, n_layers, n_neurons, device, dropout_rate=0.2):
        super(MyModel, self).__init__()
        self.device = device
        layers = []
        in_dim = 3
        for i in range(n_layers):
            out_dim = 1 if i == n_layers - 1 else n_neurons
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.Dropout(dropout_rate))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Sigmoid())
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers).to(self.device)

    def normalize(self, c1):
        return torch.clamp((c1 / 100) ** 5., min=1e-6)

    def denormalize(self, c1_normalized):
        return torch.clamp((c1_normalized ** (1. / 5.)) * 100., min=0., max=100.)

    def forward(self, x):
        x = x.to(self.device)
        gas_raw, oil_raw = x[:, 0], x[:, 1]

        gas_clipped = torch.clamp(gas_raw, min=1e-3)
        oil_clipped = torch.clamp(oil_raw, min=1e-3)
        gor         = torch.log(torch.clamp(gas_clipped / oil_clipped, min=1e-3, max=1e3))

        gas = (torch.log(gas_clipped) - 10.) / 5.
        oil = (torch.log(oil_clipped) - 5.) / 4.

        x = torch.stack([gas, oil, gor], dim=-1).to(self.device)

        c1_normalized = torch.clamp(self.mlp(x), min=1e-6, max=1 - 1e-6)
        c1 = self.denormalize(c1_normalized)
        return c1_normalized, c1

# ------------------------------ #
#        TRAIN / PREDICT         #
# ------------------------------ #

def predict(model, df):
    """
    Run model on either ['Gas','Oil'] or ['Monthly Gas','Monthly Oil'] columns.
    """
    if {'Gas','Oil'}.issubset(df.columns):
        cols = ['Gas','Oil']
    elif {'Monthly Gas','Monthly Oil'}.issubset(df.columns):
        cols = ['Monthly Gas','Monthly Oil']
    else:
        raise ValueError("Need 'Gas'+'Oil' or 'Monthly Gas'+'Monthly Oil' in df.")
    inputs = torch.tensor(df[cols].to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        _, c1 = model(inputs)
    return c1.cpu().numpy()


def find_model(data_source, threshold, radius_factor, a, b, df, df_test, comp,
               batch_size, n_layers, n_neurons, lr, n_epochs, L, seed=42):
    """
    Fit the MLP on df, evaluate MAE on df_test, with early stopping (patience=3).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check leakage between train and test (same (X_prod,Y_prod,T_prod))
    if all(col in df.columns for col in ['X_prod','Y_prod','T_prod']) and \
       all(col in df_test.columns for col in ['X_prod','Y_prod','T_prod']):
        common = df.merge(df_test, on=['X_prod','Y_prod','T_prod'], how='inner')
        print(f"Warning: {len(common)} test samples overlap with training!" if not common.empty
              else "No train/test overlap detected.")

    # Clean and types
    for dset in [df, df_test]:
        dset.dropna(subset=['Monthly Oil','Monthly Gas',comp], inplace=True)
        dset.reset_index(drop=True, inplace=True)
        dset[comp] = dset[comp].astype(float)
        dset['Monthly Gas'] = dset['Monthly Gas'].astype(float)
        dset['Monthly Oil'] = dset['Monthly Oil'].astype(float)

    dataset      = MyDataset(df,       torch.tensor(5.), 0, len(df),       comp)
    dataset_test = MyDataset(df_test,  torch.tensor(5.), 0, len(df_test),  comp)

    n_test = len(dataset_test)
    batch_size_test = (128 * (n_test // 128)) if n_test >= 128 else n_test

    dataloader      = DataLoader(dataset,      batch_size=batch_size,     shuffle=True,  drop_last=False, num_workers=4, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=True)

    loss_fn = (lambda x,y: torch.abs(x-y).mean()) if L == 1 else (lambda x,y: ((x-y)**2).mean())

    def train():
        torch.manual_seed(seed); np.random.seed(seed)
        model = MyModel(n_layers, n_neurons, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        log_dir = f'/scratch/users/pburdeau/notebooks/parameters_sweep_{data_source}'
        os.makedirs(log_dir, exist_ok=True)

        epoch_log, loss_log, mae_log = [], [], []
        best_mae, patience, patience_counter = float('inf'), 3, 0

        for epoch in range(n_epochs):
            model.train()
            losses = []
            for data_dict in dataloader:
                optimizer.zero_grad()
                x = data_dict['input'].to(device)
                _, c1 = model(x)
                loss = loss_fn(c1, data_dict['c1'].to(device))
                loss.backward(); optimizer.step()
                losses.append(loss.item())

            # Eval on test
            model.eval()
            with torch.no_grad():
                MAE = []
                for data_dict in dataloader_test:
                    x = data_dict['input'].to(device)
                    _, c1 = model(x)
                    gt = data_dict['c1'].to(device)
                    MAE.append(torch.abs(c1 - gt).mean().item())
            avg_mae = float(np.mean(MAE)); avg_loss = float(np.mean(losses))

            epoch_log.append(epoch+1); loss_log.append(avg_loss); mae_log.append(avg_mae)

            # Early stopping
            if avg_mae < best_mae:
                best_mae = avg_mae; patience_counter = 0; best_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # quick trace
            plt.figure(figsize=(10,5)); plt.clf()
            plt.subplot(1,2,1); plt.plot(epoch_log, loss_log, label='Train Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
            plt.subplot(1,2,2); plt.plot(epoch_log, mae_log,  label='Test MAE', color='red'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()
            plot_path = os.path.join(log_dir, f'parameters_swap_data_source_{data_source}_radius_factor_{radius_factor}_a_{a}_b_{b}_lr_{lr}_n_epochs_{n_epochs}_batch_size_{batch_size}_n_layers_{n_layers}_n_neurons_{n_neurons}_L_{L}.pdf')
            plt.savefig(plot_path)

        # Restore best
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass
        return model, avg_mae, min(mae_log) if mae_log else np.inf

    model, final_mae, min_mae = train()
    print(f'Final MAE: {final_mae:.4e}')
    return model, final_mae, min_mae

# ------------------------------ #
#     PRE/POST PROCESS HELPERS   #
# ------------------------------ #

def filter_threshold_production(data_source, wells_data, other_data, REL_DIFF_THRESHOLD):
    """
    Drop basins where avg production differs by more than REL_DIFF_THRESHOLD between
    wells_data and other_data (both grouped by BASIN_NAME).
    """
    if data_source == 'ghgrp':
        wells_data = wells_data[wells_data.Year >= 2015]

    gas_1 = wells_data.groupby('BASIN_NAME')['Monthly Gas'].mean()
    gas_2 = other_data.groupby('BASIN_NAME')['Monthly Gas'].mean()
    oil_1 = wells_data.groupby('BASIN_NAME')['Monthly Oil'].mean()
    oil_2 = other_data.groupby('BASIN_NAME')['Monthly Oil'].mean()

    gas_cmp = pd.DataFrame({'gas_1': gas_1, 'gas_2': gas_2}).dropna()
    oil_cmp = pd.DataFrame({'oil_1': oil_1, 'oil_2': oil_2}).dropna()

    gas_cmp['rel_diff_gas'] = abs(gas_cmp['gas_1'] - gas_cmp['gas_2']) / gas_cmp['gas_1']
    oil_cmp['rel_diff_oil'] = abs(oil_cmp['oil_1'] - oil_cmp['oil_2']) / oil_cmp['oil_1']

    valid_basins = gas_cmp[gas_cmp['rel_diff_gas'] <= REL_DIFF_THRESHOLD].index \
        .intersection(oil_cmp[oil_cmp['rel_diff_oil'] <= REL_DIFF_THRESHOLD].index)

    return other_data[other_data['BASIN_NAME'].isin(valid_basins)]


def upload_result(all_prod_indices_all, path):
    """
    Load result CSVs matching a glob pattern and merge with production voxel indices on rounded coords.
    """
    csv_files = glob.glob(os.path.join(root_path, 'out', path))
    result_all = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    # Align rounding to production-voxel coordinate rounding
    result_all['X'] = (result_all['X'] * 6.34).round(1)
    result_all['Y'] = (result_all['Y'] * 6.34).round(1)

    merged_list = []
    basins = result_all.BASIN_NAME.unique()
    with tqdm(total=len(basins)) as pbar:
        for basin in basins:
            basin_result = result_all[result_all.BASIN_NAME == basin].copy()
            basin_prod   = all_prod_indices_all[all_prod_indices_all.BASIN_NAME == basin].copy()
            if basin_result.empty or basin_prod.empty:
                pbar.update(1); continue

            basin_prod['Prod_X_grid'] = basin_prod['Prod_X_grid'].round(1)
            basin_prod['Prod_Y_grid'] = basin_prod['Prod_Y_grid'].round(1)

            merged = pd.merge(
                basin_prod, basin_result,
                left_on=['Prod_X_grid','Prod_Y_grid','Prod_T_grid'],
                right_on=['X','Y','T'],
                suffixes=('', '_result')
            )
            merged = merged.drop(columns=['X_result','Y_result','T_result','BASIN_NAME_result'], errors='ignore')
            merged = merged.rename(columns={'X':'X_prod','Y':'Y_prod','T':'T_prod'})
            merged_list.append(merged)
            pbar.update(1)

    return pd.concat(merged_list, ignore_index=True)


def filter_non_hydrocarbons(df, threshold):
    non_hc = ['HE','CO2','H2','N2','H2S','AR','O2']
    df['non_hydrocarbon_sum'] = df[non_hc].fillna(0).sum(axis=1)
    out = df[df['non_hydrocarbon_sum'] < threshold].drop(columns=['non_hydrocarbon_sum'])
    return out


def mae_weighted_variance(beta, true, pred1, pred_1_std, pred2):
    """
    Objective: MAE of blended predictor between kriging and MLP.
    """
    w_krig = 1.0
    w_pred = beta * (pred_1_std ** 2)
    yhat = (pred1 + w_pred * pred2) / (w_krig + w_pred)
    return np.mean(np.abs(true - yhat))


def predic_beta(pred1, pred_1_std, pred2, beta):
    """
    Closed-form blend with calibrated β.
    """
    w_krig = 1.0
    w_pred = beta * (pred_1_std ** 2)
    return (pred1 + w_pred * pred2) / (w_krig + w_pred)

# ------------------------------ #
#        MERGE & METRICS         #
# ------------------------------ #

def return_merged(test_block, df_test, result_all_new, result_all, result_all_nn, comp):
    """
    Merge test set with predictions (kriging-refined, simple-kriging, nn) and compute errors.
    """
    def prepare_and_analyze(df1, df2, comp):
        df1_copy = df1.copy(); df2_copy = df2.copy()

        # mark df2 rows not present in df1 by (X_prod,Y_prod,T_prod,BASIN_NAME)
        df2_unique = df2_copy.merge(df1_copy, on=['X_prod','Y_prod','T_prod','BASIN_NAME'],
                                    how='left', indicator=True)
        df2_unique = df2_unique[df2_unique['_merge'] == 'left_only'].drop(columns=['_merge'])

        # distance to nearest df2_unique point per basin
        for basin in tqdm(df1_copy['BASIN_NAME'].unique(), desc="Processing Basins"):
            sub1 = df1_copy[df1_copy['BASIN_NAME'] == basin]
            sub2 = df2_unique[df2_unique['BASIN_NAME'] == basin]
            if not sub2.empty:
                tree = KDTree(sub2[['X_prod','Y_prod','T_prod']])
                dist = tree.query(sub1[['X_prod','Y_prod','T_prod']].to_numpy())[0]
                df1_copy.loc[sub1.index, 'closest_distance'] = dist

        merged = pd.merge(df1_copy, df2_copy, on=['X_prod','Y_prod','T_prod','BASIN_NAME'])
        merged = merged[~pd.isna(merged[f'{comp}_x']) & ~pd.isna(merged[f'{comp}_y'])]

        merged['abs_error'] = np.abs(merged[f'{comp}_y'] - merged[f'{comp}_x'])
        merged['sq_error']  = (merged[f'{comp}_y'] - merged[f'{comp}_x'])**2
        merged['nse']       = ((merged[f'{comp}_y'] - merged[f'{comp}_x']) / merged['std_C1'])**2
        return merged

    merged = df_test.merge(test_block[['X_prod','Y_prod','T_prod','BASIN_NAME']],
                           on=['X_prod','Y_prod','T_prod','BASIN_NAME'], how='right', indicator=True)
    cols_to_null = ['HE','CO2','H2','N2','H2S','AR','O2','C1','C2','C3','N-C4','I-C4','N-C5','I-C5','C6+']
    merged.loc[merged['_merge'] == 'left_only', cols_to_null] = pd.NA
    merged.drop(columns=['_merge'], inplace=True)
    df1 = merged.reset_index(drop=True)

    df2_kriging        = result_all_new.reset_index(drop=True)
    df2_simple_kriging = result_all[~pd.isna(result_all[f'{comp}_predic_gor'])].reset_index(drop=True)
    df2_nn             = result_all_nn[~pd.isna(result_all_nn[f'{comp}_predic_gor'])].reset_index(drop=True)

    return (prepare_and_analyze(df1.copy(), df2_kriging, comp),
            prepare_and_analyze(df1.copy(), df2_simple_kriging, comp),
            prepare_and_analyze(df1.copy(), df2_nn, comp))

# ------------------------------ #
#      CORE APPLY PER COMP       #
# ------------------------------ #

def apply_model_comp(filter_prod, data_source, a, b,
                     df_data, df_test, df_validation,
                     test_block, validation_block, result_validation,
                     comp, n_layers, n_neurons, batch_size, n_epochs, L, lr,
                     result_all_new, result_all, result_all_nn, result_sample_2):
    """
    Train/validate/test per-component model and apply β-blend, then propagate to all outputs.
    """
    # Train on everything EXCEPT val+test blocks
    concatenated_block = pd.concat([validation_block, test_block], ignore_index=True)
    df_train = df_data.merge(concatenated_block[['X_prod','Y_prod','T_prod','BASIN_NAME']],
                             on=['X_prod','Y_prod','T_prod','BASIN_NAME'], how='left', indicator=True)
    df_train = df_train[df_train['_merge'] == 'left_only'].drop(columns=['_merge'])
    if data_source == 'usgs':
        df_train = filter_non_hydrocarbons(df_train, threshold)

    model1, _, _ = find_model(data_source, threshold, radius_factor, a, b,
                              df_train[df_train['Monthly Gas'] > 0],
                              df_test[df_test['Monthly Gas'] > 0],
                              comp, batch_size, n_layers, n_neurons, lr, n_epochs, L, seed=42)

    # Validate β on validation cells
    merged_validation = pd.merge(result_validation, df_validation,
                                 on=['X_prod','Y_prod','T_prod','BASIN_NAME','Monthly Oil','Monthly Gas'],
                                 suffixes=('_predic_k','_true')).reset_index(drop=True)
    merged_validation = merged_validation.dropna(subset=[f'{comp}_true', f'{comp}_predic_k']).reset_index(drop=True)

    if filter_prod:
        wells = []
        for basin_name in tqdm(merged_validation.BASIN_NAME.unique()):
            fpath = os.path.join(root_path, f'wells_info_prod_per_basin/{basin_name}_final.csv')
            if os.path.isfile(fpath):
                wells.append(pd.read_csv(fpath))
        if wells:
            wells_data = pd.concat(wells, ignore_index=True)
            merged_validation = filter_threshold_production(data_source, wells_data, merged_validation, 1)

    merged_validation[f'{comp}_predic_gor'] = predict(model1, merged_validation)

    initial_beta = 0.0
    res = minimize(mae_weighted_variance, initial_beta,
                   args=(merged_validation[f'{comp}_true'],
                         merged_validation[f'{comp}_predic_k'],
                         merged_validation[f'std_{comp}'],
                         merged_validation[f'{comp}_predic_gor']),
                   bounds=[(0, 100)])
    optimal_beta = float(res.x[0])

    # Re-train removing only test block, then evaluate on test
    concatenated_block = test_block
    df_train = df_data.merge(concatenated_block[['X_prod','Y_prod','T_prod','BASIN_NAME']],
                             on=['X_prod','Y_prod','T_prod','BASIN_NAME'], how='left', indicator=True)
    df_train = df_train[df_train['_merge'] == 'left_only'].drop(columns=['_merge'])
    if data_source == 'usgs':
        df_train = filter_non_hydrocarbons(df_train, threshold)

    model2, _, _ = find_model(data_source, threshold, radius_factor, a, b,
                              df_train[df_train['Monthly Gas'] > 0],
                              df_test[df_test['Monthly Gas'] > 0],
                              comp, batch_size, n_layers, n_neurons, lr, n_epochs, L, seed=42)

    # Test diagnostics
    merged_result = pd.merge(result_all, test_block,
                             on=['X_prod','Y_prod','T_prod','BASIN_NAME','Monthly Oil','Monthly Gas'],
                             suffixes=('_predic_k','_true')).reset_index(drop=True)
    merged_result = merged_result.dropna(subset=[f'{comp}_true', f'{comp}_predic_k']).reset_index(drop=True)
    if filter_prod:
        merged_result = filter_threshold_production(data_source, wells_data, merged_result, 0.9)

    merged_result[f'{comp}_predic_gor'] = predict(model2, merged_result)
    merged_result = merged_result.dropna(subset=[f'{comp}_predic_gor'])
    merged_result[comp] = predic_beta(merged_result[f'{comp}_predic_k'],
                                      merged_result[f'std_{comp}'],
                                      merged_result[f'{comp}_predic_gor'],
                                      optimal_beta)

    print('New mae: ',        np.mean(np.abs(merged_result[comp] - merged_result[f'{comp}_true'])))
    print('Kriging mae: ',    np.mean(np.abs(merged_result[f'{comp}_predic_k'] - merged_result[f'{comp}_true'])))
    print('Non-linear mae: ', np.mean(np.abs(merged_result[f'{comp}_predic_gor'] - merged_result[f'{comp}_true'])))

    # Apply to all outputs
    result_all_new[f'{comp}_predic_gor'] = predict(model2, result_all_new)
    result_all_new = result_all_new.dropna(subset=[f'{comp}_predic_gor'])
    result_all_new[comp] = predic_beta(result_all_new[comp], result_all_new[f'std_{comp}'],
                                       result_all_new[f'{comp}_predic_gor'], optimal_beta)

    result_sample_2[f'{comp}_predic_gor'] = predict(model2, result_sample_2)
    result_sample_2 = result_sample_2.dropna(subset=[f'{comp}_predic_gor'])
    result_sample_2[f'{comp}_predic_kriging'] = result_sample_2[comp]
    result_sample_2[comp] = predic_beta(result_sample_2[comp], result_sample_2[f'std_{comp}'],
                                        result_sample_2[f'{comp}_predic_gor'], optimal_beta)

    result_all[f'{comp}_predic_gor']    = predict(model2, result_all)
    result_all_nn[f'{comp}_predic_gor'] = predict(model2, result_all_nn)

    # stash β
    for df_ in (result_all_new, result_all, result_all_nn, result_sample_2):
        df_[f'beta_{comp}'] = optimal_beta

    return result_all_new, result_all, result_all_nn, result_sample_2

# ------------------------------ #
#        EXPORT MAIN LOOP        #
# ------------------------------ #

def export_results(threshold, data_source, keep_nanapis, radius_factor, filter_prod):
    """
    For each (a,b) block split:
      - load kriging result CSVs,
      - build df_data, df_test, df_validation,
      - fit/apply ML+β blend for C1 (and C2 if USGS),
      - save refined outputs.
    """
    if data_source == 'ghgrp':
        all_prod_indices_all = pd.read_csv('/scratch/users/pburdeau/notebooks/all_prod_indices_all_ghgrp.csv')
    else:
        all_prod_indices_all = pd.read_csv(f'/scratch/users/pburdeau/notebooks/all_prod_indices_all_{threshold}.csv')

    # round to match merging convention
    all_prod_indices_all['X'] = all_prod_indices_all['X'].round(1)
    all_prod_indices_all['Y'] = all_prod_indices_all['Y'].round(1)
    all_prod_indices_all['Prod_X_grid'] = all_prod_indices_all['Prod_X_grid'].round(1)
    all_prod_indices_all['Prod_Y_grid'] = all_prod_indices_all['Prod_Y_grid'].round(1)

    with tqdm(total=len([-1, 1, -1, 1])) as pbar:
        for a, b in zip([-1, 1, -1, 1], [-1, 1, 1, -1]):
            a_str, b_str = str(a), str(b)

            result_all = upload_result(
                all_prod_indices_all,
                f'result_data_source_{data_source}_a_{a_str}_b_{b_str}_basin_*_sample_1_interp_ordinary_kriging_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv'
            )
            result_sample_2 = upload_result(
                all_prod_indices_all,
                f'result_data_source_{data_source}_a_{a_str}_b_{b_str}_basin_*_sample_2_interp_ordinary_kriging_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv'
            )
            result_all_nn = upload_result(
                all_prod_indices_all,
                f'result_data_source_{data_source}_a_{a_str}_b_{b_str}_basin_*_sample_1_interp_nn_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv'
            )
            result_validation = upload_result(
                all_prod_indices_all,
                f'result_data_source_{data_source}_a_{a_str}_b_{b_str}_basin_*_sample_0_interp_ordinary_kriging_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv'
            )
            validation_block = upload_result(
                all_prod_indices_all,
                f'block_validation_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_sample_0_a_{a_str}_b_{b_str}_basin*_radius_factor_{radius_factor}.csv'
            )
            test_block = upload_result(
                all_prod_indices_all,
                f'block_test_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_sample_0_a_{a_str}_b_{b_str}_basin_*_radius_factor_{radius_factor}.csv'
            )

            # Load base data_source table
            if data_source == 'usgs':
                df_data = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_prod.csv'))
                df_data = filter_non_hydrocarbons(df_data, threshold)
            else:
                df_data = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))
                # GHGRP needs explicit *_prod columns; USGS already has them in usgs_prod.csv
                df_data['X_prod'] = df_data['X']
                df_data['Y_prod'] = df_data['Y']
                df_data['T_prod'] = df_data['T']
            

            # rounding convention
            df_data['X_prod'] = df_data['X_prod'].round(1)
            df_data['Y_prod'] = df_data['Y_prod'].round(1)

            # Build test/validation tables
            df_test = test_block[['X_prod','Y_prod','T_prod','BASIN_NAME']].merge(
                df_data, on=['X_prod','Y_prod','T_prod','BASIN_NAME']
            )
            if data_source == 'usgs':
                df_test = filter_non_hydrocarbons(df_test, threshold)

            df_validation = validation_block[['X_prod','Y_prod','T_prod','BASIN_NAME']].merge(
                df_data, on=['X_prod','Y_prod','T_prod','BASIN_NAME']
            )
            if data_source == 'usgs':
                df_validation = filter_non_hydrocarbons(df_validation, threshold)

            result_all_new = result_all.copy()

            # Apply for C1
            comp = 'C1'
            if data_source == 'usgs':
                n_layers, n_neurons, batch_size, n_epochs, L, lr = 3, 16, 128, 500, 2, 1e-4
            else:
                n_layers, n_neurons, batch_size, n_epochs, L, lr = 3, 16, 1024, 30, 2, 1e-3

            print(f'Applying model for {comp}', flush=True)
            result_all_new, result_all, result_all_nn, result_sample_2 = apply_model_comp(
                filter_prod, data_source, a_str, b_str, df_data, df_test, df_validation,
                test_block, validation_block, result_validation, comp,
                n_layers, n_neurons, batch_size, n_epochs, L, lr,
                result_all_new, result_all, result_all_nn, result_sample_2
            )

            # Optionally also C2 for USGS
            if data_source == 'usgs':
                comp = 'C2'
                print(f'Applying model for {comp}', flush=True)
                n_layers, n_neurons, batch_size, n_epochs, L, lr = 3, 16, 256, 2500, 2, 3e-5
                result_all_new, result_all, result_all_nn, result_sample_2 = apply_model_comp(
                    filter_prod, data_source, a_str, b_str, df_data, df_test, df_validation,
                    test_block, validation_block, result_validation, comp,
                    n_layers, n_neurons, batch_size, n_epochs, L, lr,
                    result_all_new, result_all, result_all_nn, result_sample_2
                )

            # Save
            result_all_new.to_csv(os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_all_new_a_{a_str}_b_{b_str}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'), index=False)
            result_all.to_csv     (os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_all_a_{a_str}_b_{b_str}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'), index=False)
            result_sample_2.to_csv(os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_sample_2_a_{a_str}_b_{b_str}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'), index=False)
            result_all_nn.to_csv  (os.path.join(out_path, f'data_source_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_result_all_nn_a_{a_str}_b_{b_str}_radius_factor_{radius_factor}_filter_{filter_prod}.csv'), index=False)

            print(f'Finished a={a_str}, b={b_str}', flush=True)
            pbar.update(1)
    return None

# ------------------------------ #
#               CLI              #
# ------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source',   type=int)
    parser.add_argument('--threshold',     type=int)
    parser.add_argument('--radius_factor', type=int)
    parser.add_argument('--filter_prod',   type=int)
    args = parser.parse_args()

    data_sources   = ['usgs', 'ghgrp']
    thresholds     = [1000, 10, 25, 50]
    radius_factors = [0.175, 0.15]
    filter_prods   = [True, False]

    data_source  = data_sources[args.data_source]
    threshold    = thresholds[args.threshold]
    radius_factor= radius_factors[args.radius_factor]
    keep_nanapis = True
    filter_prod  = filter_prods[args.filter_prod]

    export_results(threshold, data_source, keep_nanapis, radius_factor, filter_prod)
