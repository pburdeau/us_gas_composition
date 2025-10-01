#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D gridding + anisotropic covariance + interpolation (OK / SGS / NN) over basins.

Pipeline (per basin)
--------------------
1) Load produced-gas composition data (USGS or GHGRP), filter by non-HC threshold if USGS.
2) Load per-basin production (Gas/Oil) and rescale space by alpha.
3) Build (X, Y, T) grid and aggregate both production and composition to grid cells.
4) Optional block holdout selection via k-means + radius to form validation/test masks.
5) For each component:
   - Create a variogram (if enough data) or fall back to precomputed average variogram.
   - Interpolate via:
       * 'ordinary_kriging' (OK in 3D),
       * 'sgs_*' (Sequential Gaussian Simulation variants),
       * 'nn' (3D nearest neighbor), or
       * 'nn_no_time' (2D NN ignoring time).
   - Back-transform from normal scores, write results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
import skgstat as skg
from skgstat import models
import random
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import argparse
import json
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ------------------------------ #
#            CONFIG              #
# ------------------------------ #

root_path = '/scratch/users/pburdeau/data/gas_composition'
# Ensure common output dirs exist
os.makedirs(os.path.join(root_path, 'out'), exist_ok=True)
os.makedirs('/scratch/users/pburdeau/notebooks/figures_out', exist_ok=True)

# ------------------------------ #
#            GRIDDING            #
# ------------------------------ #

class Gridding:
    def prediction_grid_3d_adjusted(xmin, xmax, ymin, ymax, tmin, tmax, res):
        """Simple 3D grid with inclusive upper bounds."""
        x_range = np.arange(xmin, xmax + res, res)
        y_range = np.arange(ymin, ymax + res, res)
        t_range = np.arange(tmin, tmax + res, res)
        xx, yy, tt = np.meshgrid(x_range, y_range, t_range, indexing='ij')
        return np.stack((xx.ravel(), yy.ravel(), tt.ravel()), axis=-1)

    def prediction_grid_3d(xmin, xmax, ymin, ymax, tmin, tmax, res):
        """
        Generate a 3D prediction grid with exclusive upper bounds (linspace).
        Returns array of [x, y, t] coords.
        """
        cols = int(np.ceil((xmax - xmin) / res))
        rows = int(np.ceil((ymax - ymin) / res))
        depths = int(np.ceil((tmax - tmin) / res))
        x = np.linspace(xmin, xmax, num=cols, endpoint=False)
        y = np.linspace(ymin, ymax, num=rows, endpoint=False)
        t = np.linspace(tmin, tmax, num=depths, endpoint=False)
        xx, yy, tt = np.meshgrid(x, y, t, indexing='ij')
        return np.stack((xx.ravel(), yy.ravel(), tt.ravel()), axis=-1)

    def grid_data_3d(df, xx, yy, tt, zz, res_space, res_time, mean_agg=True):
        """
        Aggregate values to a regular (X,Y,T) grid with mean or sum.
        Returns (df_grid, grid_matrix, grid_shape).
        """
        df = df.rename(columns={xx: "X", yy: "Y", zz: "Z", tt: "T"})
        xmin, xmax = df['X'].min(), df['X'].max()
        ymin, ymax = df['Y'].min(), df['Y'].max()
        tmin, tmax = df['T'].min(), df['T'].max()

        x_range = np.arange(xmin, xmax + res_space, res_space)
        y_range = np.arange(ymin, ymax + res_space, res_space)
        t_range = np.arange(tmin, tmax + res_time, res_time)
        grid_coord = np.meshgrid(x_range, y_range, t_range, indexing='ij')

        np_data = df[['X', 'Y', 'T', 'Z']].to_numpy()
        origin = np.array([xmin, ymin, tmin])
        resolution = np.array([res_space, res_space, res_time])
        grid_indices = np.rint((np_data[:, :3] - origin) / resolution).astype(int)

        grid_shape = (len(x_range), len(y_range), len(t_range))
        grid_sum = np.zeros(grid_shape)
        grid_count = np.zeros(grid_shape)

        for i in range(np_data.shape[0]):
            x_idx, y_idx, t_idx = grid_indices[i, :3]
            val = np_data[i, 3]
            if x_idx < grid_shape[0] and y_idx < grid_shape[1] and t_idx < grid_shape[2]:
                if not np.isnan(val):
                    grid_sum[x_idx, y_idx, t_idx] += val
                    grid_count[x_idx, y_idx, t_idx] += 1

        if mean_agg:
            with np.errstate(invalid='ignore', divide='ignore'):
                grid_matrix = np.divide(grid_sum, grid_count)
        else:
            grid_matrix = grid_sum

        flat_grid_coords = np.column_stack([g.ravel() for g in grid_coord])
        df_grid = pd.DataFrame(flat_grid_coords, columns=['X', 'Y', 'T'])
        df_grid['Z'] = grid_matrix.ravel()
        df_grid['Count'] = grid_count.ravel()
        return df_grid, grid_matrix, grid_shape

# ------------------------------ #
#       NEAREST NEIGHBORS        #
# ------------------------------ #

class NearestNeighbor:
    def nearest_neighbor_search_3d(radius, num_points, loc, data):
        """
        Return up to num_points nearest neighbors within radius in 3D (X,Y,T).
        """
        locx, locy, loct = loc
        data = data.copy()
        centered = np.column_stack((data['X'].values - locx,
                                    data['Y'].values - locy,
                                    data['T'].values - loct))
        data["dist"] = np.sqrt(np.sum(centered ** 2, axis=1))
        within = data[data["dist"] < radius]
        nearest = within.sort_values("dist").head(num_points)
        return nearest[['X', 'Y', 'T', 'Z', 'U']].values

    def nearest_neighbor_secondary_3d(loc, data):
        """
        Find the single nearest secondary data point (X,Y,T) to loc.
        """
        locx, locy, loct = loc
        data = data.copy()
        centered = np.column_stack((data['X'].values - locx,
                                    data['Y'].values - locy,
                                    data['T'].values - loct))
        data["dist"] = np.sqrt(np.sum(centered ** 2, axis=1))
        data = data.sort_values('dist', ascending=True).reset_index(drop=True)
        return data.iloc[0][['X', 'Y', 'T', 'Z']].values

# ------------------------------ #
#        ROTATION MATRIX 3D      #
# ------------------------------ #

def make_rotation_matrix_3d(azimuth, dip, major_range, minor_range):
    """
    3D rotation+scaling for anisotropy: rotate by dip then azimuth; scale by 1/range.
    """
    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)
    R_azimuth = np.array([[np.cos(azimuth_rad), -np.sin(azimuth_rad), 0],
                          [np.sin(azimuth_rad),  np.cos(azimuth_rad), 0],
                          [0, 0, 1]])
    R_dip = np.array([[ np.cos(dip_rad), 0, np.sin(dip_rad)],
                      [ 0, 1, 0],
                      [-np.sin(dip_rad), 0, np.cos(dip_rad)]])
    S = np.diag([1 / major_range, 1 / minor_range, 1])
    return R_azimuth @ R_dip @ S

# ------------------------------ #
#           COVARIANCE           #
# ------------------------------ #

class Covariance:
    def covar(effective_lag, sill, nug, vtype):
        """
        Covariance from a variogram model (exp/gauss/spherical).
        Effective lag is normalized (range = 1).
        """
        if vtype.lower() == 'exponential':
            c = (sill - nug) * np.exp(-3 * effective_lag)
        elif vtype.lower() == 'gaussian':
            c = (sill - nug) * np.exp(-3 * np.square(effective_lag))
        elif vtype.lower() == 'spherical':
            c = sill - nug - 1.5 * effective_lag + 0.5 * np.power(effective_lag, 3)
            c[effective_lag > 1] = sill - 1
        else:
            raise AttributeError("vtype must be 'Exponential', 'Gaussian', or 'Spherical'")
        return c

    def make_covariance_matrix_3d(coord, vario, rotation_matrix):
        """Full covariance matrix C( coord, coord )."""
        nug = vario[1]; sill = vario[4]; vtype = vario[5]
        mat = coord @ rotation_matrix
        eff = pairwise_distances(mat, mat)
        return Covariance.covar(eff, sill, nug, vtype)

    def make_covariance_array_3d(coord1, coord2, vario, rotation_matrix):
        """Covariance vector c( coord1, coord2 )."""
        nug = vario[1]; sill = vario[4]; vtype = vario[5]
        mat1 = coord1 @ rotation_matrix
        mat2 = (coord2.reshape(-1, 3)) @ rotation_matrix
        eff = np.sqrt(np.square(mat1 - mat2).sum(axis=1))
        return Covariance.covar(eff, sill, nug, vtype)

# ------------------------------ #
#         INTERPOLATION          #
# ------------------------------ #

class Interpolation:
    def okrige_3d(prediction_grid, df, xx, yy, tt, zz, uncertainty_col, num_points, vario, radius, quiet=False):
        """
        Ordinary kriging in 3D (X,Y,T) with Lagrange multiplier.
        Returns (estimates, kriging variances).
        """
        azimuth = vario[0]; major_range = vario[2]; minor_range = vario[3]; dip = 0
        rotation_matrix = make_rotation_matrix_3d(azimuth, dip, major_range, minor_range)
        df = df.rename(columns={xx:"X", yy:"Y", tt:"T", zz:"Z", uncertainty_col:"U"})
        var_1 = vario[4]
        est_ok = np.zeros(len(prediction_grid))
        var_ok = np.zeros(len(prediction_grid))
        _iterator = tqdm(prediction_grid, position=0, leave=True) if not quiet else prediction_grid

        for z, predxyz in enumerate(_iterator):
            test_idx = np.sum((df[['X','Y']].values == predxyz[:2]).all(axis=1) &
                              (np.abs(df['T'].values - predxyz[2]) < 1e-5))
            if test_idx == 0:
                nearest = NearestNeighbor.nearest_neighbor_search_3d(
                    radius, num_points, prediction_grid[z], df[['X','Y','T','Z','U']]
                )
                if len(nearest) == 0:
                    est_ok[z] = np.nan; var_ok[z] = np.nan
                    continue

                norm_vals = nearest[:, -2]
                local_mean = np.mean(norm_vals)
                xyzt_val = nearest[:, :-2]
                n = len(nearest)

                if n > 0:
                    cov_mat = np.zeros((n+1, n+1))
                    cov_mat[:n, :n] = Covariance.make_covariance_matrix_3d(xyzt_val, vario, rotation_matrix)
                    cov_mat[n, :n] = 1; cov_mat[:n, n] = 1

                    cov_vec = np.zeros(n+1)
                    cov_vec[:n] = Covariance.make_covariance_array_3d(
                        xyzt_val, np.tile(prediction_grid[z], (n,1)), vario, rotation_matrix
                    )
                    cov_vec[n] = 1

                    k_weights, res, rank, s = np.linalg.lstsq(cov_mat, cov_vec, rcond=None)
                    est_ok[z] = local_mean + np.sum(k_weights[:n] * (norm_vals - local_mean))
                    var_ok[z] = var_1 - np.sum(k_weights[:n] * cov_vec[:n])
                    var_ok[var_ok < 0] = 0
            else:
                est_ok[z] = df.loc[(df[['X','Y']].values == predxyz[:2]).all(axis=1) &
                                   (np.abs(df['T'].values - predxyz[2]) < 1e-5), 'Z'].iloc[0]
                var_ok[z] = 0
        return est_ok, var_ok

    def nearest_neighbor_estimation(prediction_grid, df1, xx1, yy1, tt1, zz1, radius, quiet=False):
        """
        3D nearest neighbor (X,Y,T) with radius. Returns (est, 0).
        """
        est_nn = np.zeros(len(prediction_grid)); var_nn = np.zeros(len(prediction_grid))
        df1 = df1.rename(columns={xx1:"X", yy1:"Y", tt1:"T", zz1:"Z"})
        _iterator = prediction_grid if quiet else tqdm(prediction_grid, position=0, leave=True)
        for i, (x,y,t) in enumerate(_iterator):
            d = np.sqrt((df1["X"]-x)**2 + (df1["Y"]-y)**2 + (df1["T"]-t)**2)
            within = d <= radius
            if within.any():
                nearest_idx = d.argmin()
                est_nn[i] = df1.iloc[nearest_idx]["Z"]
            else:
                est_nn[i] = np.nan
            var_nn[i] = 0
        return est_nn, var_nn

    def nearest_neighbor_estimation_no_time(prediction_grid, df1, xx1, yy1, tt1, zz1, radius, quiet=False):
        """
        2D nearest neighbor ignoring time (uses only X,Y).
        Note: 'radius' is not used to filter; behavior matches your original code.
        """
        est_nn = np.zeros(len(prediction_grid)); var_nn = np.zeros(len(prediction_grid))
        df1 = df1.rename(columns={xx1:"X", yy1:"Y", tt1:"T", zz1:"Z"})
        _iterator = prediction_grid if quiet else tqdm(prediction_grid, position=0, leave=True)
        for i, (x,y,t) in enumerate(_iterator):
            d = np.sqrt((df1["X"]-x)**2 + (df1["Y"]-y)**2)
            # Original behavior: no radius mask; pick global nearest
            nearest_idx = d.argmin()
            est_nn[i] = df1.iloc[nearest_idx]["Z"]
            var_nn[i] = 0
        return est_nn, var_nn

    def okrige_sgs_3d_old(prediction_grid, df, xx, yy, tt, zz, uncertainty_col, num_points, vario, radius, quiet=False):
        """
        Sequential Gaussian Simulation (legacy version). Uses OK at each step to sample.
        """
        azimuth = vario[0]; major_range = vario[2]; minor_range = vario[3]
        rotation_matrix = make_rotation_matrix_3d(azimuth, 0, major_range, minor_range)
        # Clear/rename
        if 'X' in df.columns and xx != 'X': df = df.drop(columns=['X'])
        if 'Y' in df.columns and yy != 'Y': df = df.drop(columns=['Y'])
        if 'T' in df.columns and tt != 'T': df = df.drop(columns=['T'])
        if 'Z' in df.columns and zz != 'Z': df = df.drop(columns=['Z'])
        if 'U' in df.columns and uncertainty_col != 'U': df = df.drop(columns=['U'])
        df = df.rename(columns={xx:"X", yy:"Y", tt:"T", zz:"Z", uncertainty_col:"U"})

        xyindex = np.arange(len(prediction_grid))
        random.shuffle(xyindex)
        var_1 = vario[4]
        sgs = np.zeros(shape=len(prediction_grid))

        for idx, predxyz in enumerate(tqdm(prediction_grid, position=0, leave=True, disable=quiet)):
            z = xyindex[idx]
            test_idx = np.sum(prediction_grid[z][:2] == df[['X','Y']].values, axis=1) & \
                       (np.abs(df['T'].values - predxyz[2]) < 1e-5)
            if np.sum(test_idx) == 0:
                nearest = NearestNeighbor.nearest_neighbor_search_3d(
                    radius, num_points, prediction_grid[z], df[['X','Y','T','Z','U']]
                )
                if nearest.shape[0] == 0:
                    sgs[z] = np.nan
                    continue

                norm_vals = nearest[:, -2]
                xyzt_val = nearest[:, :-2]
                local_mean = np.mean(norm_vals)
                n = len(nearest)

                cov_mat = np.zeros((n+1, n+1))
                cov_mat[:n, :n] = Covariance.make_covariance_matrix_3d(xyzt_val, vario, rotation_matrix)
                cov_mat[n, :n] = 1; cov_mat[:n, n] = 1

                cov_vec = np.zeros(n+1)
                cov_vec[:n] = Covariance.make_covariance_array_3d(
                    xyzt_val, np.tile(prediction_grid[z], (n,1)), vario, rotation_matrix
                )
                cov_vec[n] = 1

                k_weights, res, rank, s = np.linalg.lstsq(cov_mat, cov_vec, rcond=None)
                est = local_mean + np.sum(k_weights[:n] * (norm_vals - local_mean))
                var = var_1 - np.sum(k_weights[:n] * cov_vec[:n])
                var = np.abs(var)
                sgs[z] = np.random.normal(est, np.sqrt(var), 1)
            else:
                sgs[z] = df['Z'].values[np.where(test_idx)[0][0]]
        return sgs

    def okrige_sgs_3d(prediction_grid, df, xx, yy, tt, zz, uncertainty_col, num_points, vario, radius, quiet=False):
        """
        Sequential Gaussian Simulation (current). Adds simulated node back into conditioning set.
        """
        azimuth = vario[0]; major_range = vario[2]; minor_range = vario[3]; dip = 0
        rotation_matrix = make_rotation_matrix_3d(azimuth, dip, major_range, minor_range)
        df = df.rename(columns={xx:"X", yy:"Y", tt:"T", zz:"Z", uncertainty_col:"U"})
        var_1 = vario[4]
        sgs = np.zeros(len(prediction_grid))
        _iterator = tqdm(prediction_grid, position=0, leave=True) if not quiet else prediction_grid

        for z, predxyz in enumerate(_iterator):
            test_idx = np.sum((df[['X','Y']].values == predxyz[:2]).all(axis=1) &
                              (np.abs(df['T'].values - predxyz[2]) < 1e-5))
            if test_idx == 0:
                nearest = NearestNeighbor.nearest_neighbor_search_3d(
                    radius, num_points, prediction_grid[z], df[['X','Y','T','Z','U']]
                )
                if len(nearest) == 0:
                    sgs[z] = np.nan
                    continue

                norm_vals = nearest[:, -2]
                local_mean = np.mean(norm_vals)
                xyzt_val = nearest[:, :-2]
                n = len(nearest)

                cov_mat = np.zeros((n+1, n+1))
                cov_mat[:n, :n] = Covariance.make_covariance_matrix_3d(xyzt_val, vario, rotation_matrix)
                cov_mat[n, :n] = 1; cov_mat[:n, n] = 1

                cov_vec = np.zeros(n+1)
                cov_vec[:n] = Covariance.make_covariance_array_3d(
                    xyzt_val, np.tile(prediction_grid[z], (n,1)), vario, rotation_matrix
                )
                cov_vec[n] = 1

                k_weights, res, rank, s = np.linalg.lstsq(cov_mat, cov_vec, rcond=None)
                est = local_mean + np.sum(k_weights[:n] * (norm_vals - local_mean))
                var = var_1 - np.sum(k_weights[:n] * cov_vec[:n])
                var = np.abs(var)
                sgs[z] = np.random.normal(est, np.sqrt(var), 1)
            else:
                sgs[z] = df.loc[(df[['X','Y']].values == predxyz[:2]).all(axis=1) &
                                (np.abs(df['T'].values - predxyz[2]) < 1e-5), 'Z'].iloc[0]

            # Append simulated node to conditioning set
            coords = prediction_grid[z:z+1, :]
            df = pd.concat([df, pd.DataFrame({'X':[coords[0,0]], 'Y':[coords[0,1]], 'T':[coords[0,2]], 'Z':[sgs[z]]})],
                           sort=False)
        return sgs

# ------------------------------ #
#        GRID + AGG HELPERS      #
# ------------------------------ #

def create_grid(df, res_space, res_time):
    """Create ij-mesh grid across (X, Y, T) extents and return (grid, params)."""
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

def aggregate(df, list_zz, grid_params, mean_agg=True, weight_col=None):
    """
    Aggregate variables list_zz onto the provided (X,Y,T) grid.
    Supports weighted or unweighted mean; otherwise sum if mean_agg=False.
    """
    xmin, xmax, ymin, ymax, tmin, tmax, res_space, res_time = grid_params
    x_range = np.arange(xmin, xmax + res_space, res_space)
    y_range = np.arange(ymin, ymax + res_space, res_space)
    t_range = np.arange(tmin, tmax + res_time, res_time)
    grid_coord = np.meshgrid(x_range, y_range, t_range, indexing='ij')

    df = df[(df.X >= xmin) & (df.X <= xmax) &
            (df.Y >= ymin) & (df.Y <= ymax) &
            (df['T'] >= tmin) & (df['T'] <= tmax)].reset_index(drop=True)

    flat_grid_coords = np.column_stack([g.ravel() for g in grid_coord])
    df_grid = pd.DataFrame(flat_grid_coords, columns=['X', 'Y', 'T'])

    for zz in list_zz:
        list_cols = ['X', 'Y', 'T', zz]
        if weight_col:
            list_cols.append(weight_col)

        np_data = df[list_cols].to_numpy().astype(float)
        origin = np.array([xmin, ymin, tmin])
        resolution = np.array([res_space, res_space, res_time])
        idx = np.rint(((np_data[:, :3] - origin) / resolution)).astype(int)

        shape = (len(x_range), len(y_range), len(t_range))
        grid_sum = np.zeros(shape)
        grid_count = np.zeros(shape)

        if weight_col:
            grid_weighted_sum = np.zeros(shape)
            grid_weight_sum = np.zeros(shape)

        for i in range(np_data.shape[0]):
            x_idx, y_idx, t_idx = idx[i, :3]
            if x_idx < shape[0] and y_idx < shape[1] and t_idx < shape[2]:
                if weight_col:
                    w = np_data[i, 4]
                    grid_weighted_sum[x_idx, y_idx, t_idx] += np_data[i, 3] * w
                    grid_weight_sum[x_idx, y_idx, t_idx] += w
                else:
                    grid_sum[x_idx, y_idx, t_idx] += np_data[i, 3]
                    grid_count[x_idx, y_idx, t_idx] += 1

        if mean_agg:
            with np.errstate(invalid='ignore', divide='ignore'):
                if weight_col:
                    grid_matrix = np.divide(grid_weighted_sum, grid_weight_sum)
                else:
                    grid_matrix = np.divide(grid_sum, grid_count)
        else:
            grid_matrix = grid_weighted_sum if weight_col else grid_sum

        df_grid[zz] = grid_matrix.ravel()

    return df_grid

# ------------------------------ #
#      DIAGNOSTIC PLOTTING       #
# ------------------------------ #

def plot_histograms(df_grid, zz, nzz, maxlag, sample, interp_method, keep_nanapis, threshold, data_source):
    """Compare original vs normalized distributions for a component in grid."""
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.hist(df_grid[zz], facecolor='red', bins=50, alpha=0.2, edgecolor='black')
    plt.xlabel(zz); plt.ylabel('Frequency'); plt.title('Original'); plt.grid(True)

    plt.subplot(122)
    plt.hist(df_grid[nzz], facecolor='red', bins=50, alpha=0.2, edgecolor='black')
    plt.xlabel(nzz); plt.ylabel('Frequency'); plt.title('Normalized'); plt.grid(True)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    figures_path = '/scratch/users/pburdeau/notebooks/figures_out/'
    os.makedirs(figures_path, exist_ok=True)
    fname = (f'hist_data_source_{data_source}_maxlag_{maxlag}_comp_{zz}_'
             f'keep_nanapis_{keep_nanapis}_interp_method_{interp_method}_'
             f'sample_{sample}_threshold_{threshold}.pdf')
    plt.savefig(os.path.join(figures_path, fname), bbox_inches='tight')
    plt.close()

def plot_variogram(V1, comp, basin, maxlag, sample, interp_method, keep_nanapis, threshold, data_source):
    """Plot experimental + exponential (and compute others but only plot exp)."""
    xdata = V1.bins
    ydata = V1.experimental

    V1.model = 'exponential'
    V2 = V1; V2.model = 'gaussian'
    V3 = V1; V3.model = 'spherical'

    xi = np.linspace(0, xdata[-1], 100)
    y_exp = [models.exponential(h, V1.parameters[0], V1.parameters[1], V1.parameters[2]) for h in xi]
    y_gauss = [models.gaussian(h, V2.parameters[0], V2.parameters[1], V2.parameters[2]) for h in xi]
    y_sph = [models.spherical(h, V3.parameters[0], V3.parameters[1], V3.parameters[2]) for h in xi]

    fig = plt.figure()
    plt.plot(xdata / 1000, ydata, 'og', label="Experimental variogram")
    plt.plot(xi / 1000, y_exp, 'b-', label='Exponential variogram')
    plt.title(f'Isotropic variogram for {comp} with maxlag {maxlag}')
    plt.xlabel('Lag [km]'); plt.ylabel('Semivariance'); plt.legend(loc='lower right')

    variograms_path = os.path.join(root_path, 'out', 'variograms')
    os.makedirs(variograms_path, exist_ok=True)
    # (Saving was commented in your original; left unchanged.)

# ------------------------------ #
#       VARIOGRAM CREATION       #
# ------------------------------ #

def create_variogram(data_single, comp, basin, n_lags, maxlag, sample, interp_method, keep_nanapis, threshold, data_source):
    """
    Build per-basin variogram for a component after normal-score transform.
    Falls back to average variogram parameters if insufficient variability or fitting fails.
    """
    data = data_single[comp].values.reshape(-1, 1)
    uncertainties = data_single['std_' + comp].values.reshape(-1, 1)

    nst_trans_comp = QuantileTransformer(n_quantiles=500, output_distribution='normal').fit(data)
    data_single['norm_' + comp] = nst_trans_comp.transform(data)
    data_single['norm_std_' + comp] = nst_trans_comp.transform(uncertainties)

    if len(np.unique(data_single['norm_' + comp])) <= 1 or np.std(data_single['norm_' + comp]) < 1e-6:
        print(f"Insufficient variability for {comp} in {basin}. Using average variogram.")
        output_dir = os.path.join(root_path, 'average_variograms',
                                  f'{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}')
        variogram_file = os.path.join(output_dir, f'{comp}_variogram.json')
        try:
            with open(variogram_file, 'r') as f:
                v = json.load(f)
            azimuth = v.get("azimuth", 0); nugget = v.get("nugget", 0.1)
            major_range = v.get("major_range", 1.0); minor_range = v.get("minor_range", 1.0)
            sill = v.get("sill", 1.0); vtype = v.get("type", 'Exponential')
        except FileNotFoundError:
            print(f"Fallback variogram file not found for {comp} in {basin}. Using default parameters.")
            azimuth, nugget, major_range, minor_range, sill, vtype = 0, 0.1, 1.0, 1.0, 1.0, 'Exponential'
        return nst_trans_comp, azimuth, nugget, major_range, minor_range, sill, vtype

    coords = data_single[['X', 'Y', 'T']].values
    values = data_single['norm_' + comp]

    try:
        V1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, maxlag=maxlag, normalize=False)
        plot_variogram(V1, comp, basin, maxlag, sample, interp_method, keep_nanapis, threshold, data_source)
        azimuth = 0
        nugget = V1.parameters[2]
        major_range = V1.parameters[0]
        minor_range = V1.parameters[0]
        sill = V1.parameters[1]
        vtype = 'Exponential'
        print(f'Created variogram in {basin} for {comp}.')
    except RuntimeError as e:
        print(f"Variogram creation failed for {comp} in {basin}: {e}. Using average variogram.")
        output_dir = os.path.join(root_path, 'average_variograms',
                                  f'{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}')
        variogram_file = os.path.join(output_dir, f'{comp}_variogram.json')
        try:
            with open(variogram_file, 'r') as f:
                v = json.load(f)
            azimuth = v.get("azimuth", 0); nugget = v.get("nugget", 0.1)
            major_range = v.get("major_range", 1.0); minor_range = v.get("minor_range", 1.0)
            sill = v.get("sill", 1.0); vtype = v.get("type", 'Exponential')
        except FileNotFoundError:
            print(f"Fallback variogram file not found for {comp} in {basin}. Using default parameters.")
            azimuth, nugget, major_range, minor_range, sill, vtype = 0, 0.1, 1.0, 1.0, 1.0, 'Exponential'

    return nst_trans_comp, azimuth, nugget, major_range, minor_range, sill, vtype

# ------------------------------ #
#      PER-COMP INTERPOLATION    #
# ------------------------------ #

def interpolate_comp(comp, grid, data_on_grid, basin, n_lags, maxlag, k, rad,
                     sample, interp_method, keep_nanapis, threshold, data_source):
    """
    Interpolate one component on the provided grid using chosen method,
    then back-transform normal scores and attach std estimates.
    """
    print('Start interpolating ' + comp + '...')
    data_single = data_on_grid[~pd.isna(data_on_grid[comp])].reset_index(drop=True)
    data_single = data_single[data_single[comp] > 0].reset_index(drop=True)
    print(f'{len(data_single)} initial datapoints for {comp} in {basin}')
    if len(data_single) > 0:
        if len(data_single) >= 50 and len(np.unique(data_single[comp])) > 1 and np.std(data_single[comp]) > 1e-6:
            print(f'Trying to create variogram in {basin} for {comp}')
            nst_trans_comp, azimuth, nugget, major_range, minor_range, sill, vtype = create_variogram(
                data_single, comp, basin, n_lags, maxlag, sample, interp_method, keep_nanapis, threshold, data_source)
        else:
            output_dir = os.path.join(root_path, 'average_variograms',
                                      f'{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}')
            variogram_file = os.path.join(output_dir, f'{comp}_variogram.json')
            with open(variogram_file, 'r') as f:
                v = json.load(f)
            azimuth = v.get("azimuth"); nugget = v.get("nugget")
            major_range = v.get("major_range"); minor_range = v.get("minor_range")
            sill = v.get("sill"); vtype = v.get("type")
            data = data_single[comp].values.reshape(-1, 1)
            uncertainties = data_single['std_' + comp].values.reshape(-1, 1)
            nst_trans_comp = QuantileTransformer(n_quantiles=500, output_distribution='normal').fit(data)
            data_single['norm_' + comp] = nst_trans_comp.transform(data)
            data_single['norm_std_' + comp] = nst_trans_comp.transform(uncertainties)

        vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

        if interp_method.startswith('sgs') and interp_method in [f'sgs_{i}' for i in range(1, 11)]:
            data_out = Interpolation.okrige_sgs_3d(grid, data_single, 'X', 'Y', 'T', 'norm_' + comp, 'norm_std_' + comp,
                                                   k, vario, rad)
            var_out = np.zeros(len(data_out))
            print('Done interpolating with SGS for ' + comp + ' in ' + basin + '...')
        elif interp_method == 'nn':
            print('Start interpolating with nearest neighbor ' + comp + 'in ' + basin + '...')
            data_out, var_out = Interpolation.nearest_neighbor_estimation(
                grid, data_single, 'X', 'Y', 'T', 'norm_' + comp, rad
            )
            print('Done interpolating with nearest neighbor ' + comp + 'in ' + basin + '...')
        elif interp_method == 'ordinary_kriging':
            data_out, var_out = Interpolation.okrige_3d(
                grid, data_single, 'X', 'Y', 'T', 'norm_' + comp, 'norm_std_' + comp, k, vario, rad
            )
        elif interp_method == 'nn_no_time':
            data_out, var_out = Interpolation.nearest_neighbor_estimation_no_time(
                grid, data_single, 'X', 'Y', 'T', 'norm_' + comp, rad
            )

        var_out[var_out < 0] = 0
        std_out = np.sqrt(var_out)

        data_out = data_out.reshape(-1, 1)
        std_out = std_out.reshape(-1, 1)
        data_out = nst_trans_comp.inverse_transform(data_out)
        std_out = nst_trans_comp.inverse_transform(std_out)
        std_out = std_out - np.nanmin(std_out)
    else:
        data_out = [np.nan for _ in range(len(grid))]
        std_out = [np.nan for _ in range(len(grid))]

    data_on_grid = data_on_grid.copy()
    data_on_grid[comp] = data_out
    data_on_grid['std_' + comp] = std_out
    return data_on_grid

# ------------------------------ #
#  INTERPOLATE ALL COMPONENTS    #
# ------------------------------ #

def interpolate_all_components(components, basin, df_data, df_prod, res_space, res_time,
                               alpha, k, rad, maxlag, n_lags, interp_method,
                               frac, sample, a, b, keep_nanapis, threshold, data_source, radius_factor):
    """
    Build grid, mask cells by Gas>0, optional block holdouts, interpolate all components,
    and attach Gas/Oil back to output.
    """
    sorted_components = components
    grid, grid_params = create_grid(df_prod, res_space, res_time)
    prod_on_grid = aggregate(df_prod, ['Gas'], grid_params, mean_agg=False)
    data_on_grid = aggregate(df_data, components, grid_params, mean_agg=True)
    prod_oil_on_grid = aggregate(df_prod, ['Oil'], grid_params, mean_agg=False)
    data_on_grid = data_on_grid[prod_on_grid.Gas > 0].reset_index(drop=True)

    # Block holdout selection via 4 k-means centers
    if a != 2 and data_on_grid[~pd.isna(data_on_grid.C1)][['X','Y']].drop_duplicates().shape[0] > 5:
        np.random.seed(42)
        nb_clusters = 4
        data_on_grid_not_na = data_on_grid[~pd.isna(data_on_grid.C1)]
        coordinates = data_on_grid_not_na[['X','Y']].to_numpy()
        kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
        data_on_grid_not_na['Cluster'] = kmeans.fit_predict(coordinates)
        centers = kmeans.cluster_centers_
        x_range = np.max(data_on_grid_not_na['X']) - np.min(data_on_grid_not_na['X'])
        y_range = np.max(data_on_grid_not_na['Y']) - np.min(data_on_grid_not_na['Y'])
        radius = radius_factor * min(x_range, y_range)
        distances = cdist(centers, centers, metric='euclidean')
        used_for_selector_0 = set(); used_for_selector_1 = set()
        selector_pairs = []

        while len(selector_pairs) < nb_clusters:
            selector_0_idx = next(idx for idx in range(nb_clusters) if idx not in used_for_selector_0)
            available_selector_1 = [idx for idx in range(nb_clusters) if idx not in used_for_selector_1 and idx != selector_0_idx]
            if not available_selector_1:
                available_selector_1 = [idx for idx in range(nb_clusters) if idx != selector_0_idx]
            if not available_selector_1:
                raise ValueError(f"No more unused centers for validation/test in {basin} (thr={threshold}, keep_nanapis={keep_nanapis}, sample={sample}, a={a}, b={b}, radius_factor={radius_factor})")

            farthest_idx = max(available_selector_1, key=lambda idx: distances[selector_0_idx, idx])
            used_for_selector_0.add(selector_0_idx); used_for_selector_1.add(farthest_idx)

            d0 = np.sqrt((data_on_grid['X']-centers[selector_0_idx][0])**2 + (data_on_grid['Y']-centers[selector_0_idx][1])**2)
            selector_0 = d0 <= radius
            d1 = np.sqrt((data_on_grid['X']-centers[farthest_idx][0])**2 + (data_on_grid['Y']-centers[farthest_idx][1])**2)
            selector_1 = (d1 <= radius) & ~selector_0
            selector_pairs.append(((selector_0, selector_0_idx), (selector_1, farthest_idx)))

        def map_to_index(a_list, b_list):
            pairs = list(zip(a_list, b_list))
            return {pair: idx for idx, pair in enumerate(pairs)}

        i = map_to_index([-1, 1, -1, 1], [-1, 1, 1, -1])[(a, b)]
        selector_0, selector_0_center_idx = selector_pairs[i][0]
        selector_1, selector_1_center_idx = selector_pairs[i][1]
        selector = selector_0 | selector_1

        # Export chosen holdouts
        data_on_grid_not_selected_0 = data_on_grid[selector_0].copy()
        data_on_grid_not_selected_0['BASIN_NAME'] = [basin] * len(data_on_grid_not_selected_0)
        data_on_grid_not_selected_0.to_csv(os.path.join(root_path, 'out',
                                f'block_validation_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_sample_{sample}_a_{a}_b_{b}_basin_{basin}_radius_factor_{radius_factor}.csv'), index=False)

        data_on_grid_not_selected_1 = data_on_grid[selector_1].copy()
        data_on_grid_not_selected_1['BASIN_NAME'] = [basin] * len(data_on_grid_not_selected_1)
        data_on_grid_not_selected_1.to_csv(os.path.join(root_path, 'out',
                                f'block_test_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_sample_{sample}_a_{a}_b_{b}_basin_{basin}_radius_factor_{radius_factor}.csv'), index=False)

        # Holdout visualization (circle, inputs, cluster center)
        plt.figure(figsize=(16, 12))
        ax = plt.gca()
        circle = Circle(centers[selector_0_center_idx], radius, color='#93a09d', alpha=0.12, label='Holdout area')
        ax.add_patch(circle)
        data_on_grid_not_na = data_on_grid[~pd.isna(data_on_grid.C1)]
        plt.scatter(data_on_grid_not_na['X'], data_on_grid_not_na['Y'], color='#d2c9b6', s=42, label='Input set')
        data_on_grid_not_selected_0_not_na = data_on_grid[selector_0 & ~pd.isna(data_on_grid.C1)]
        plt.scatter(data_on_grid_not_selected_0_not_na['X'], data_on_grid_not_selected_0_not_na['Y'],
                    color='#93a09d', s=42, label='Holdout set')
        plt.scatter(centers[selector_0_center_idx,0], centers[selector_0_center_idx,1],
                    color='black', s=200, marker='X', label='Cluster center')
        ax.set_aspect('equal', adjustable='box')
        plt.axis('off'); plt.tight_layout()
        figures_path = '/scratch/users/pburdeau/notebooks/figures_out/'
        os.makedirs(figures_path, exist_ok=True)
        plt.savefig(os.path.join(figures_path,
            f'plot_test_block_no_legend_{data_source}_basin_{basin}_keep_nanapis_{keep_nanapis}_sample_{sample}_threshold_{threshold}_radius_factor_{radius_factor}_a_{a}_b_{b}.png'))
        plt.show()

        # Legend-only figure
        fig_legend = plt.figure(figsize=(12, 9))
        legend_handles = [
            Line2D([0],[0], marker='o', color='w', label='Input set', markerfacecolor='#d2c9b6', markersize=20),
            Line2D([0],[0], marker='o', color='w', label='Holdout set', markerfacecolor='#93a09d', markersize=20),
            Line2D([0],[0], marker='X', color='w', label='Cluster center', markerfacecolor='black', markersize=20),
            Patch(facecolor='#93a09d', edgecolor='none', alpha=0.12, label='Holdout area')
        ]
        fig_legend.legend(handles=legend_handles, loc='center', fontsize=24, frameon=True)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
        plt.axis('off')
        plt.savefig(os.path.join(figures_path,
            f'legend_plot_test_block_{data_source}_keep_nanapis_{keep_nanapis}_sample_{sample}_threshold_{threshold}_radius_factor_{radius_factor}_a_{a}_b_{b}.png'))
        plt.show()

        # Overview with all circles and pairs
        plt.figure(figsize=(16, 12))
        plt.scatter(data_on_grid_not_na['X'], data_on_grid_not_na['Y'], color='#d2c9b6', s=24, label='Input sets')
        ax = plt.gca()
        for idx, center in enumerate(centers):
            ax.add_patch(Circle(center, radius, color='#93a09d', alpha=0.07, label='Holdout areas' if idx == 0 else None))

        j = 0
        for (sel0, _c0), (sel1, _c1) in selector_pairs:
            d0 = data_on_grid[sel0]; d0 = d0[~pd.isna(d0.C1)]
            plt.scatter(d0['X'], d0['Y'], s=24, color='#93a09d')
            d1 = data_on_grid[sel1]; d1 = d1[~pd.isna(d1.C1)]
            plt.scatter(d1['X'], d1['Y'], s=24, color='#93a09d')
            if j == 3:
                plt.scatter(d1['X'], d1['Y'], s=24, color='#93a09d', label='Holdout sets')
            j += 1

        ax.set_aspect('equal', adjustable='box')
        plt.scatter(centers[:,0], centers[:,1], color='black', s=200, marker='X', label='Cluster centers')
        plt.legend(loc='upper right', fontsize=24, markerscale=1.5, frameon=True)
        plt.axis('off'); plt.tight_layout(); plt.show()

        # Apply holdout mask to remove block(s) from training depending on sample setting
        selector_0, _ = selector_pairs[i][0]
        selector_1, _ = selector_pairs[i][1]
        selector = selector_0 | selector_1
        if sample == 1:  # remove only test for total
            selector = selector_1
        if not sample == 2:  # sample=2 means keep all
            data_on_grid.loc[selector, components] = np.nan

    # Build final grid + attach Gas/Oil
    df_grid = pd.DataFrame({'X': grid[:,0], 'Y': grid[:,1], 'T': grid[:,2]})
    prod_oil_on_grid = prod_oil_on_grid[prod_on_grid.Gas > 0].reset_index(drop=True)
    df_grid = df_grid[prod_on_grid.Gas > 0].reset_index(drop=True)
    grid = df_grid.values
    prod_on_grid = prod_on_grid[prod_on_grid.Gas > 0].reset_index(drop=True)

    for comp in components:
        data_on_grid['std_' + comp] = 0  # initial uncertainties (USGS assumption)

    # Interpolate per component
    for i, comp in enumerate(components, start=1):
        data_on_grid = interpolate_comp(
            comp, grid, data_on_grid, basin, n_lags, maxlag, k, rad, sample,
            interp_method, keep_nanapis, threshold, data_source
        )
        print(f'Component {i} on {len(sorted_components)} done.')

    data_on_grid['Gas'] = prod_on_grid['Gas']
    data_on_grid['Oil'] = prod_oil_on_grid['Oil']
    data_on_grid['check_sum'] = data_on_grid[components].sum(axis=1)
    return data_on_grid

# ------------------------------ #
#           DRIVER               #
# ------------------------------ #

def my_function(data_source, basin, interp_method, sample, keep_nanapis, threshold, radius_factor):
    """
    Load data, set params, interpolate all components for one basin, and save outputs.
    """
    # ---- Load composition data ----
    if data_source == 'usgs':
        if keep_nanapis:
            data = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_with_nanapis.csv'))
        else:
            data = pd.read_csv(os.path.join(root_path, 'usgs', 'usgs_processed_without_nanapis.csv'))

        def filter_non_hydrocarbons(df, threshold):
            non_hydrocarbons = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2']
            df['non_hydrocarbon_sum'] = df[non_hydrocarbons].fillna(0).sum(axis=1)
            out = df[df['non_hydrocarbon_sum'] < threshold].drop(columns=['non_hydrocarbon_sum'])
            return out

        data = filter_non_hydrocarbons(data, threshold)
    else:
        data = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))

    # ---- Load per-basin production ----
    wells_data = pd.read_csv(os.path.join(root_path, f'wells_info_prod_per_basin/{basin}_final.csv'))

    # ---- Parameters ----
    frac = 0.9
    alpha = 6.34
    n_lags = 10
    k = 100
    res_space = 2000 / alpha
    res_time = 1825  # 5 years
    maxlag = 100000 / alpha
    rad = maxlag

    # ---- Prepare dataframes ----
    df_data = data.reset_index(drop=True)
    df_data = df_data[df_data.BASIN_NAME == basin].reset_index(drop=True)
    df_data['X'] = df_data.X / alpha
    df_data['Y'] = df_data.Y / alpha

    df_prod = wells_data[wells_data.BASIN_NAME == basin].reset_index(drop=True)
    df_prod = df_prod.rename(columns={'Monthly Oil':'Oil', 'Monthly Gas':'Gas'})
    df_prod = df_prod[df_prod.Gas > 0].reset_index(drop=True)
    df_prod['X'] = df_prod.X / alpha
    df_prod['Y'] = df_prod.Y / alpha
    df_prod = df_prod[~pd.isna(df_prod['Gas'])].reset_index(drop=True)

    components = ['HE','CO2','H2','N2','H2S','AR','O2','C1','C2','C3','N-C4','I-C4','N-C5','I-C5','C6+']
    components = [c for c in components if c in df_data.columns]

    # ---- Interpolation ----
    if interp_method.startswith('sgs') and interp_method in [f'sgs_{i}' for i in range(1, 11)]:
        a, b = 2, 2
        data_on_grid = interpolate_all_components(
            components, basin, df_data, df_prod, res_space, res_time, alpha, k, rad,
            maxlag, n_lags, interp_method, frac, sample, a, b, keep_nanapis, threshold,
            data_source, radius_factor
        )
        data_on_grid['BASIN_NAME'] = basin
        out_csv = os.path.join(root_path, 'out',
            f'test_result_data_source_{data_source}_a_{a}_b_{b}_basin_{basin}_sample_{sample}_interp_{interp_method}_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')
        data_on_grid.to_csv(out_csv, index=False)
    else:
        for a, b in zip([-1, 1, -1, 1], [-1, 1, 1, -1]):
            data_on_grid = interpolate_all_components(
                components, basin, df_data, df_prod, res_space, res_time, alpha, k, rad,
                maxlag, n_lags, interp_method, frac, sample, a, b, keep_nanapis, threshold,
                data_source, radius_factor
            )
            data_on_grid['BASIN_NAME'] = basin
            out_csv = os.path.join(root_path, 'out',
                f'result_data_source_{data_source}_a_{a}_b_{b}_basin_{basin}_sample_{sample}_interp_{interp_method}_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv')
            data_on_grid.to_csv(out_csv, index=False)

# ------------------------------ #
#             MAIN               #
# ------------------------------ #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--basin_id', type=int)
    parser.add_argument('--sample_id', type=int)
    parser.add_argument('--interp_method_id', type=int)
    parser.add_argument('--data_source', type=int)
    parser.add_argument('--threshold', type=int)
    parser.add_argument('--keep_nanapis', type=int)
    parser.add_argument('--radius_factor', type=int)
    arguments = parser.parse_args()

    basins_many_datapoints = [
        'Gulf Coast Basin (LA, TX)', 'San Joaquin Basin', 'Powder River Basin', 'Chautauqua Platform',
        'Arkoma Basin', 'San Juan Basin', 'Green River Basin', 'Paradox Basin', 'Piceance Basin',
        'Appalachian Basin (Eastern Overthrust Area)', 'Palo Duro Basin', 'East Texas Basin',
        'Denver Basin', 'Arkla Basin', 'South Oklahoma Folded Belt', 'Bend Arch', 'Permian Basin',
        'Fort Worth Syncline', 'Wind River Basin', 'Uinta Basin', 'Williston Basin', 'Las Animas Arch',
        'Appalachian Basin', 'Anadarko Basin'
    ]

    bools = [0, 1, 2]
    sample = bools[arguments.sample_id]

    interp_methods = ['nn', 'ordinary_kriging', 'sgs_1', 'sgs_2', 'sgs_3', 'sgs_4', 'sgs_5']
    interp_method = interp_methods[arguments.interp_method_id]

    data_sources = ['usgs', 'ghgrp']
    data_source = data_sources[arguments.data_source]

    thresholds = [1000, 25, 50, 10]
    threshold = thresholds[arguments.threshold]

    keep_nanapiss = [True, False]
    keep_nanapis = keep_nanapiss[arguments.keep_nanapis]

    radius_factors = [0.15, 0.10]
    radius_factor = radius_factors[arguments.radius_factor]

    # Parallelize across basins (honor SLURM allocation if available)
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", 16))
    basin_groups = np.array_split(basins_many_datapoints, num_cpus)

    Parallel(n_jobs=num_cpus)(
        delayed(my_function)(data_source, basin, interp_method, sample, keep_nanapis, threshold, radius_factor)
        for group in basin_groups for basin in group
    )
