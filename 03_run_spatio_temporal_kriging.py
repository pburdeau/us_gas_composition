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


root_path = '/scratch/users/pburdeau/data/gas_composition'

class Gridding:

    def prediction_grid_3d_adjusted(xmin, xmax, ymin, ymax, tmin, tmax, res):
        x_range = np.arange(xmin, xmax + res, res)
        y_range = np.arange(ymin, ymax + res, res)
        t_range = np.arange(tmin, tmax + res, res)

        xx, yy, tt = np.meshgrid(x_range, y_range, t_range, indexing='ij')
        prediction_grid_xyz = np.stack((xx.ravel(), yy.ravel(), tt.ravel()), axis=-1)

        return prediction_grid_xyz

    def prediction_grid_3d(xmin, xmax, ymin, ymax, tmin, tmax, res):
        """
        Generate a 3D prediction grid.

        Parameters
        ----------
            xmin, xmax : float, int
                Minimum and maximum x extents.
            ymin, ymax : float, int
                Minimum and maximum y extents.
            tmin, tmax : float, int
                Minimum and maximum t extents (third dimension).
            res : float, int
                Grid cell resolution for x, y, and t dimensions.

        Returns
        -------
            prediction_grid_xyz : numpy.ndarray
                Array of [x, y, t] coordinates for the prediction grid.
        """
        # Calculate the number of grid cells in each dimension
        cols = int(np.ceil((xmax - xmin) / res))
        rows = int(np.ceil((ymax - ymin) / res))
        depths = int(np.ceil((tmax - tmin) / res))

        # Generate grid points
        x = np.linspace(xmin, xmax, num=cols, endpoint=False)
        y = np.linspace(ymin, ymax, num=rows, endpoint=False)
        t = np.linspace(tmin, tmax, num=depths, endpoint=False)

        # Create meshgrid for 3D coordinates
        xx, yy, tt = np.meshgrid(x, y, t, indexing='ij')

        # Flatten and combine into a single array of 3D coordinates
        prediction_grid_xyz = np.stack((xx.ravel(), yy.ravel(), tt.ravel()), axis=-1)

        return prediction_grid_xyz

    def grid_data_3d(df, xx, yy, tt, zz, res_space, res_time, mean_agg=True):
        """
        Grid conditioning data in 3D (x, y, t dimensions).

        Parameters
        ----------
            df : pandas DataFrame
                Dataframe of conditioning data and coordinates.
            xx : string
                Column name for x coordinates of input data frame.
            yy : string
                Column name for y coordinates of input data frame.
            zz : string
                Column name for z values (or data variable) of input data frame.
            tt : string
                Column name for t coordinates (third dimension) of input data frame.
            res : float, int
                Grid cell resolution in x, y, and t dimensions.

        Returns
        -------
            df_grid : pandas DataFrame
                Dataframe of gridded data.
            grid_matrix : numpy.ndarray
                3D matrix of gridded data (x, y, t).
            shape : tuple
                Shape of the grid_matrix (rows, cols, depths).
        """
        df = df.rename(columns={xx: "X", yy: "Y", zz: "Z", tt: "T"})

        xmin, xmax = df['X'].min(), df['X'].max()
        ymin, ymax = df['Y'].min(), df['Y'].max()
        tmin, tmax = df['T'].min(), df['T'].max()

        # Create 3D grid coordinates
        x_range = np.arange(xmin, xmax + res_space, res_space)
        y_range = np.arange(ymin, ymax + res_space, res_space)
        t_range = np.arange(tmin, tmax + res_time, res_time)
        grid_coord = np.meshgrid(x_range, y_range, t_range, indexing='ij')

        # Convert DataFrame to NumPy array for processing
        np_data = df[['X', 'Y', 'T', 'Z']].to_numpy()
        origin = np.array([xmin, ymin, tmin])
        resolution = np.array([res_space, res_space, res_time])

        # Calculate grid indices for each data point
        grid_indices = np.rint((np_data[:, :3] - origin) / resolution).astype(int)

        # Initialize 3D aggregation arrays
        grid_shape = (len(x_range), len(y_range), len(t_range))
        grid_sum = np.zeros(grid_shape)
        grid_count = np.zeros(grid_shape)

        # Aggregate data into the grid
        for i in range(np_data.shape[0]):
            x_idx, y_idx, t_idx = grid_indices[i, :3]
            value = np_data[i, 3]  # Current value to aggregate
            if x_idx < grid_shape[0] and y_idx < grid_shape[1] and t_idx < grid_shape[2]:
                if not np.isnan(value):  # Only aggregate non-NaN values
                    grid_sum[x_idx, y_idx, t_idx] += value
                    grid_count[x_idx, y_idx, t_idx] += 1

        if mean_agg == True:
            # Calculate average values where count is not zero
            with np.errstate(invalid='ignore', divide='ignore'):
                grid_matrix = np.divide(grid_sum, grid_count)
        #             grid_matrix = np.nan_to_num(grid_matrix)  # Replace NaNs with 0
        else:
            grid_matrix = grid_sum

        # Flatten the grid for DataFrame representation
        flat_grid_coords = np.column_stack([g.ravel() for g in grid_coord])
        flat_grid_matrix = grid_matrix.ravel()
        flat_grid_count = grid_count.ravel()

        # Construct DataFrame
        df_grid = pd.DataFrame(flat_grid_coords, columns=['X', 'Y', 'T'])
        df_grid['Z'] = flat_grid_matrix
        df_grid['Count'] = flat_grid_count

        return df_grid, grid_matrix, grid_shape


####################################

# Nearest neighbor octant search

####################################

class NearestNeighbor:

    def nearest_neighbor_search_3d(radius, num_points, loc, data):
        """
        Nearest neighbor search extended to 3D (including time or a third spatial dimension)

        Parameters
        ----------
            radius : int, float
                search radius
            num_points : int
                number of points to search for
            loc : numpy.ndarray
                coordinates for grid cell of interest (now including x, y, and t)
            data : pandas DataFrame
                data with columns ['X', 'Y', 'T', 'Z','U'] for coordinates and values

        Returns
        -------
            near : numpy.ndarray
                nearest neighbors in 3D space
        """

        # Extract location coordinates
        locx, locy, loct = loc

        # Copy data to avoid modifying original DataFrame
        data = data.copy()

        # Calculate centered array for distance calculation in 3D
        centered_array = np.column_stack((
            data['X'].values - locx,
            data['Y'].values - locy,
            data['T'].values - loct
        ))

        # Calculate Euclidean distance in 3D
        data["dist"] = np.sqrt(np.sum(centered_array ** 2, axis=1))

        # Filter data within the search radius
        data_within_radius = data[data["dist"] < radius]

        # Sort data by distance, ascending
        nearest_data = data_within_radius.sort_values("dist").head(num_points)
        # Return nearest neighbors as a numpy array
        near = nearest_data[['X', 'Y', 'T', 'Z', 'U']].values
        return near

    def nearest_neighbor_secondary_3d(loc, data):
        """
        Find the nearest neighbor secondary data point to a grid cell of interest in 3D space.

        Parameters
        ----------
            loc : numpy.ndarray
                3D coordinates for the grid cell of interest, including x, y, and t dimensions.
            data : pandas DataFrame
                Secondary data with columns ['X', 'Y', 'T', 'Z'] representing spatial coordinates
                and the data value.

        Returns
        -------
            nearest_second : numpy.ndarray
                Coordinates and value of the nearest neighbor to the secondary data in 3D space.
        """

        locx, locy, loct = loc  # Unpack the 3D location coordinates

        # Copy the data to avoid modifying the original DataFrame
        data = data.copy()

        # Calculate the centered array for distance calculation in 3D
        centered_array = np.column_stack((
            data['X'].values - locx,
            data['Y'].values - locy,
            data['T'].values - loct
        ))

        # Calculate Euclidean distance in 3D
        data["dist"] = np.sqrt(np.sum(centered_array ** 2, axis=1))

        # Sort data by distance in ascending order and reset index
        data = data.sort_values('dist', ascending=True).reset_index(drop=True)

        # Select the nearest data point
        nearest_second = data.iloc[0][['X', 'Y', 'T', 'Z']].values
        return nearest_second


#########################

# Rotation Matrix

#########################

def make_rotation_matrix_3d(azimuth, dip, major_range, minor_range):
    """
    Make rotation matrix for 3D spatial data to accommodate anisotropy.

    Parameters
    ----------
        azimuth : int, float
            Azimuth angle in degrees from north, clockwise, for the major direction.
        dip : int, float
            Dip angle in degrees, down from horizontal, for the tilt of the major direction.
        major_range : int, float
            Range parameter of the variogram in the major direction.
        minor_range : int, float
            Range parameter of the variogram in the minor direction, orthogonal to azimuth on the horizontal plane.

    Returns
    -------
        rotation_matrix_3d : numpy.ndarray
            3x3 rotation matrix used to perform coordinate transformations in 3D space.
    """
    # Convert angles from degrees to radians
    azimuth_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)

    # Rotation matrix for azimuth
    R_azimuth = np.array([
        [np.cos(azimuth_rad), -np.sin(azimuth_rad), 0],
        [np.sin(azimuth_rad), np.cos(azimuth_rad), 0],
        [0, 0, 1]
    ])

    # Rotation matrix for dip
    R_dip = np.array([
        [np.cos(dip_rad), 0, np.sin(dip_rad)],
        [0, 1, 0],
        [-np.sin(dip_rad), 0, np.cos(dip_rad)]
    ])

    # Scaling matrix
    S = np.diag([1 / major_range, 1 / minor_range, 1])  # Assuming no scaling in the vertical ('t') direction

    # Composite rotation matrix: first apply dip, then azimuth rotation, and finally scaling
    rotation_matrix_3d = np.dot(np.dot(R_azimuth, R_dip), S)

    return rotation_matrix_3d


###########################

# Covariance functions

###########################

class Covariance:

    def covar(effective_lag, sill, nug, vtype):
        """
        Compute covariance

        Parameters
        ----------
            effective_lag : int, float
                lag distance that is normalized to a range of 1
            sill : int, float
                sill of variogram
            nug : int, float
                nugget of variogram
            vtype : string
                type of variogram model (Exponential, Gaussian, or Spherical)
        Raises
        ------
        AtrributeError : if vtype is not 'Exponential', 'Gaussian', or 'Spherical'

        Returns
        -------
            c : numpy.ndarray
                covariance
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
        """
        Adapted to handle 3D coordinates.
        """
        # Extract variogram parameters
        nug = vario[1]
        sill = vario[4]
        vtype = vario[5]

        # Apply rotation matrix to 3D coordinates
        mat = np.matmul(coord, rotation_matrix)

        # Calculate pairwise distances in 3D
        effective_lag = pairwise_distances(mat, mat)

        # Calculate covariance matrix using the adapted covariance function for 3D
        covariance_matrix = Covariance.covar(effective_lag, sill, nug, vtype)

        return covariance_matrix

    def make_covariance_array_3d(coord1, coord2, vario, rotation_matrix):
        """
        Adapted to handle 3D coordinates.
        """
        # Extract variogram parameters
        nug = vario[1]
        sill = vario[4]
        vtype = vario[5]

        # Apply rotation matrix to 3D coordinates
        mat1 = np.matmul(coord1, rotation_matrix)
        mat2 = np.matmul(coord2.reshape(-1, 3), rotation_matrix)  # Ensure coord2 is reshaped for 3D

        # Calculate effective lag in 3D
        effective_lag = np.sqrt(np.square(mat1 - mat2).sum(axis=1))

        # Calculate covariance array using the adapted covariance function for 3D
        covariance_array = Covariance.covar(effective_lag, sill, nug, vtype)

        return covariance_array


######################################

# Kriging Function

######################################
class Interpolation:
    def okrige_3d(prediction_grid, df, xx, yy, tt, zz, uncertainty_col, num_points, vario, radius, quiet=False):
        """
        Ordinary kriging interpolation in 3D space (x, y, t).

        Parameters are extended to include:
            tt : string
                column name for t coordinates of input data frame.

        Other parameters remain the same as in the original function.
        """
        # Unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        dip = 0
        # Assume make_rotation_matrix can handle 3D, including time adjustment if necessary
        rotation_matrix = make_rotation_matrix_3d(azimuth, dip, major_range, minor_range)
        print(rotation_matrix.shape)
        df = df.rename(columns={xx: "X", yy: "Y", tt: "T", zz: "Z", uncertainty_col: "U"})
        var_1 = vario[4]
        est_ok = np.zeros(len(prediction_grid))
        var_ok = np.zeros(len(prediction_grid))

        _iterator = tqdm(prediction_grid, position=0, leave=True) if not quiet else prediction_grid

        for z, predxyz in enumerate(_iterator):
            # Ensure proper broadcasting
            test_idx = np.sum(
                (df[['X', 'Y']].values == predxyz[:2]).all(axis=1) & (np.abs(df['T'].values - predxyz[2]) < 1e-5))

            if test_idx == 0:

                # Placeholder for nearest neighbor search logic
                nearest = NearestNeighbor.nearest_neighbor_search_3d(radius, num_points, prediction_grid[z],
                                                                     df[['X', 'Y', 'T', 'Z', 'U']])
                if len(nearest) == 0:
                    # Handle the case where no neighbors are found
                    est_ok[z] = np.nan  # or use some other placeholder value
                    var_ok[z] = np.nan  # or use some other placeholder value
                    continue

                norm_data_val = nearest[:, -2]
                local_mean = np.mean(norm_data_val)
                xyzt_val = nearest[:, :-2]
                new_num_pts = len(nearest)

                # Proceed with covariance matrix calculation only if nearest is not empty
                if new_num_pts > 0:
                    covariance_matrix = np.zeros((new_num_pts + 1, new_num_pts + 1))
                    covariance_matrix[:new_num_pts, :new_num_pts] = Covariance.make_covariance_matrix_3d(xyzt_val,
                                                                                                         vario,
                                                                                                         rotation_matrix)
                    covariance_matrix[new_num_pts, :new_num_pts] = 1
                    covariance_matrix[:new_num_pts, new_num_pts] = 1
                    # for i in range(new_num_pts):
                    #     covariance_matrix[i, i] += nearest[i, -1] ** 2  # Assuming the last column is the uncertainty

                    covariance_array = np.zeros(new_num_pts + 1)
                    covariance_array[:new_num_pts] = Covariance.make_covariance_array_3d(xyzt_val,
                                                                                         np.tile(prediction_grid[z],
                                                                                                 (new_num_pts, 1)),
                                                                                         vario, rotation_matrix)
                    covariance_array[new_num_pts] = 1

                    k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond=None)

                    est_ok[z] = local_mean + np.sum(k_weights[:new_num_pts] * (norm_data_val - local_mean))
                    var_ok[z] = var_1 - np.sum(k_weights[:new_num_pts] * covariance_array[:new_num_pts])
                    var_ok[var_ok < 0] = 0  # Ensure variance is non-negative
            else:
                est_ok[z] = df.loc[(df[['X', 'Y']].values == predxyz[:2]).all(axis=1) & (
                        np.abs(df['T'].values - predxyz[2]) < 1e-5), 'Z'].iloc[0]
                var_ok[z] = 0

        return est_ok, var_ok

    def nearest_neighbor_estimation(prediction_grid, df1, xx1, yy1, tt1, zz1, radius, quiet=False):
        """
        Simple nearest neighbor estimation for 3D data. This function uses the primary dataset to
        estimate values at prediction locations based solely on the nearest neighbor's value.

        Parameters
        ----------
            prediction_grid : numpy.ndarray
                The grid locations where predictions are to be made, as an array of [x, y, t] coordinates.
            df1 : pandas.DataFrame
                The primary dataset containing observed values and their coordinates.
            xx1, yy1, tt1, zz1 : str
                Column names in df1 for x, y, t coordinates, and the observed values, respectively.
            df2, xx2, yy2, tt2, zz2 :
                Parameters related to a secondary dataset, not used in this nearest neighbor approach but kept for compatibility.
            num_points : int
                The number of nearest neighbors to consider (1 for simple nearest neighbor).
            vario : list
                Variogram parameters, not used in this approach but kept for compatibility.
            radius : float
                The search radius within which to look for nearest neighbors.
            corrcoef : float
                Correlation coefficient, not used in this approach but kept for compatibility.
            quiet : bool
                If True, suppresses the tqdm progress bar.

        Returns
        -------
            est_nn : numpy.ndarray
                Estimated values at prediction locations based on nearest neighbor.
            var_nn : numpy.ndarray
                Variance of the estimates, set to zero in this implementation since nearest neighbor doesn't provide variance.
        """
        # Initialize arrays for estimated values and their variance
        est_nn = np.zeros(len(prediction_grid))
        var_nn = np.zeros(len(prediction_grid))

        # Convert column names for consistency
        df1 = df1.rename(columns={xx1: "X", yy1: "Y", tt1: "T", zz1: "Z"})

        # Iterator setup
        _iterator = prediction_grid if quiet else tqdm(prediction_grid, position=0, leave=True)

        for i, (x, y, t) in enumerate(_iterator):
            # Calculate distances from current prediction point to all points in df1
            distances = np.sqrt((df1["X"] - x) ** 2 + (df1["Y"] - y) ** 2 + (df1["T"] - t) ** 2)

            # Find the index of the nearest neighbor within the specified radius
            within_radius = distances <= radius
            if within_radius.any():
                # nearest_idx = distances[within_radius].argmin()
                nearest_idx = distances.argmin()
                est_nn[i] = df1.iloc[nearest_idx]["Z"]
            else:
                # If no neighbor is found within the radius, set estimation to NaN or some placeholder
                est_nn[i] = np.nan

            # Variance is not defined for a simple nearest neighbor estimate in this implementation
            var_nn[i] = 0

        return est_nn, var_nn

    def nearest_neighbor_estimation_no_time(prediction_grid, df1, xx1, yy1, tt1, zz1, radius, quiet=False):
        """
        Simple nearest neighbor estimation for 3D data. This function uses the primary dataset to
        estimate values at prediction locations based solely on the nearest neighbor's value.

        Parameters
        ----------
            prediction_grid : numpy.ndarray
                The grid locations where predictions are to be made, as an array of [x, y, t] coordinates.
            df1 : pandas.DataFrame
                The primary dataset containing observed values and their coordinates.
            xx1, yy1, tt1, zz1 : str
                Column names in df1 for x, y, t coordinates, and the observed values, respectively.
            df2, xx2, yy2, tt2, zz2 :
                Parameters related to a secondary dataset, not used in this nearest neighbor approach but kept for compatibility.
            num_points : int
                The number of nearest neighbors to consider (1 for simple nearest neighbor).
            vario : list
                Variogram parameters, not used in this approach but kept for compatibility.
            radius : float
                The search radius within which to look for nearest neighbors.
            corrcoef : float
                Correlation coefficient, not used in this approach but kept for compatibility.
            quiet : bool
                If True, suppresses the tqdm progress bar.

        Returns
        -------
            est_nn : numpy.ndarray
                Estimated values at prediction locations based on nearest neighbor.
            var_nn : numpy.ndarray
                Variance of the estimates, set to zero in this implementation since nearest neighbor doesn't provide variance.
        """
        # Initialize arrays for estimated values and their variance
        est_nn = np.zeros(len(prediction_grid))
        var_nn = np.zeros(len(prediction_grid))

        # Convert column names for consistency
        df1 = df1.rename(columns={xx1: "X", yy1: "Y", tt1: "T", zz1: "Z"})

        # Iterator setup
        _iterator = prediction_grid if quiet else tqdm(prediction_grid, position=0, leave=True)

        for i, (x, y, t) in enumerate(_iterator):
            # Calculate distances from current prediction point to all points in df1
            distances = np.sqrt((df1["X"] - x) ** 2 + (df1["Y"] - y) ** 2 )

            # Find the index of the nearest neighbor within the specified radius
            within_radius = distances
            if within_radius.any():
                # nearest_idx = distances[within_radius].argmin()
                nearest_idx = distances.argmin()
                est_nn[i] = df1.iloc[nearest_idx]["Z"]
            else:
                # If no neighbor is found within the radius, set estimation to NaN or some placeholder
                est_nn[i] = np.nan

            # Variance is not defined for a simple nearest neighbor estimate in this implementation
            var_nn[i] = 0

        return est_nn, var_nn

    def okrige_sgs_3d_old(prediction_grid, df, xx, yy, tt, zz, uncertainty_col, num_points, vario, radius, quiet=False):
        """
        Sequential Gaussian simulation using ordinary kriging in 3D space (x, y, t).
        """

        # Unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        rotation_matrix = make_rotation_matrix_3d(azimuth, 0, major_range, minor_range)  # Handle 3D rotation

        # Prepare DataFrame
        if 'X' in df.columns and xx != 'X':
            df = df.drop(columns=['X'])
        if 'Y' in df.columns and yy != 'Y':
            df = df.drop(columns=['Y'])
        if 'T' in df.columns and tt != 'T':
            df = df.drop(columns=['T'])
        if 'Z' in df.columns and zz != 'Z':
            df = df.drop(columns=['Z'])
        if 'U' in df.columns and uncertainty_col != 'U':
            df = df.drop(columns=['U'])

        df = df.rename(columns={xx: "X", yy: "Y", tt: "T", zz: "Z", uncertainty_col: "U"})
        xyindex = np.arange(len(prediction_grid))
        random.shuffle(xyindex)
        var_1 = vario[4]
        sgs = np.zeros(shape=len(prediction_grid))

        for idx, predxyz in enumerate(tqdm(prediction_grid, position=0, leave=True, disable=quiet)):
            z = xyindex[idx]
            test_idx = np.sum(prediction_grid[z][:2] == df[['X', 'Y']].values, axis=1) & (
                        np.abs(df['T'].values - predxyz[2]) < 1e-5)

            if np.sum(test_idx) == 0:
                # Gather nearest neighbor points including uncertainty
                nearest = NearestNeighbor.nearest_neighbor_search_3d(radius, num_points,
                                                                     prediction_grid[z], df[['X', 'Y', 'T', 'Z', 'U']])

                if nearest.shape[0] == 0:  # Check if no neighbors found
                    sgs[z] = np.nan  # or some other placeholder value
                    continue

                norm_data_val = nearest[:, -2]  # Z values
                xyzt_val = nearest[:, :-2]  # X, Y, T values
                local_mean = np.mean(norm_data_val)
                new_num_pts = len(nearest)

                # Covariance between data
                covariance_matrix = np.zeros(shape=((new_num_pts + 1, new_num_pts + 1)))
                covariance_matrix[:new_num_pts, :new_num_pts] = Covariance.make_covariance_matrix_3d(xyzt_val, vario,
                                                                                                     rotation_matrix)
                covariance_matrix[new_num_pts, :new_num_pts] = 1
                covariance_matrix[:new_num_pts, new_num_pts] = 1

                # Set up Right Hand Side (covariance between data and unknown)
                covariance_array = np.zeros(shape=(new_num_pts + 1))
                k_weights = np.zeros(shape=(new_num_pts + 1))
                covariance_array[:new_num_pts] = Covariance.make_covariance_array_3d(xyzt_val,
                                                                                     np.tile(prediction_grid[z],
                                                                                             (new_num_pts, 1)),
                                                                                     vario, rotation_matrix)
                covariance_array[new_num_pts] = 1

                # Solve the system of equations for kriging weights
                k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond=None)
                est = local_mean + np.sum(k_weights[:new_num_pts] * (norm_data_val - local_mean))
                var = var_1 - np.sum(k_weights[:new_num_pts] * covariance_array[:new_num_pts])
                var = np.abs(var)

                # Generate random sample for SGS
                sgs[z] = np.random.normal(est, np.sqrt(var), 1)
            else:
                sgs[z] = df['Z'].values[np.where(test_idx)[0][0]]

            # coords = prediction_grid[z:z + 1, :]
            # df = pd.concat(
            #     [df, pd.DataFrame({'X': [coords[0, 0]], 'Y': [coords[0, 1]], 'T': [coords[0, 2]], 'Z': [sgs[z]]})],
            #     sort=False)

        return sgs

    def okrige_sgs_3d(prediction_grid, df, xx, yy, tt, zz, uncertainty_col, num_points, vario, radius, quiet=False):
        """
        Ordinary kriging interpolation in 3D space (x, y, t).

        Parameters are extended to include:
            tt : string
                column name for t coordinates of input data frame.

        Other parameters remain the same as in the original function.
        """
        # Unpack variogram parameters
        azimuth = vario[0]
        major_range = vario[2]
        minor_range = vario[3]
        dip = 0
        # Assume make_rotation_matrix can handle 3D, including time adjustment if necessary
        rotation_matrix = make_rotation_matrix_3d(azimuth, dip, major_range, minor_range)
        print(rotation_matrix.shape)
        df = df.rename(columns={xx: "X", yy: "Y", tt: "T", zz: "Z", uncertainty_col: "U"})
        var_1 = vario[4]
        sgs = np.zeros(len(prediction_grid))

        _iterator = tqdm(prediction_grid, position=0, leave=True) if not quiet else prediction_grid

        for z, predxyz in enumerate(_iterator):
            # Ensure proper broadcasting
            test_idx = np.sum(
                (df[['X', 'Y']].values == predxyz[:2]).all(axis=1) & (np.abs(df['T'].values - predxyz[2]) < 1e-5))

            if test_idx == 0:

                # Placeholder for nearest neighbor search logic
                nearest = NearestNeighbor.nearest_neighbor_search_3d(radius, num_points, prediction_grid[z],
                                                                     df[['X', 'Y', 'T', 'Z', 'U']])
                if len(nearest) == 0:
                    # Handle the case where no neighbors are found
                    sgs[z] = np.nan
                    continue

                norm_data_val = nearest[:, -2]
                local_mean = np.mean(norm_data_val)
                xyzt_val = nearest[:, :-2]
                new_num_pts = len(nearest)

                # Proceed with covariance matrix calculation only if nearest is not empty
                if new_num_pts > 0:
                    covariance_matrix = np.zeros((new_num_pts + 1, new_num_pts + 1))
                    covariance_matrix[:new_num_pts, :new_num_pts] = Covariance.make_covariance_matrix_3d(xyzt_val,
                                                                                                         vario,
                                                                                                         rotation_matrix)
                    covariance_matrix[new_num_pts, :new_num_pts] = 1
                    covariance_matrix[:new_num_pts, new_num_pts] = 1
                    # for i in range(new_num_pts):
                    #     covariance_matrix[i, i] += nearest[i, -1] ** 2  # Assuming the last column is the uncertainty

                    covariance_array = np.zeros(new_num_pts + 1)
                    covariance_array[:new_num_pts] = Covariance.make_covariance_array_3d(xyzt_val,
                                                                                         np.tile(prediction_grid[z],
                                                                                                 (new_num_pts, 1)),
                                                                                         vario, rotation_matrix)
                    covariance_array[new_num_pts] = 1

                    k_weights, res, rank, s = np.linalg.lstsq(covariance_matrix, covariance_array, rcond=None)
                    est = local_mean + np.sum(k_weights[:new_num_pts] * (norm_data_val - local_mean))
                    var = var_1 - np.sum(k_weights[:new_num_pts] * covariance_array[:new_num_pts])
                    var = np.abs(var)

                    # Generate random sample for SGS
                    sgs[z] = np.random.normal(est, np.sqrt(var), 1)
            #                     sgs[z] = est
            else:
                #                 sgs[z] = df['Z'].values[np.where(test_idx)[0][0]]
                sgs[z] = df.loc[(df[['X', 'Y']].values == predxyz[:2]).all(axis=1) & (
                        np.abs(df['T'].values - predxyz[2]) < 1e-5), 'Z'].iloc[0]

            coords = prediction_grid[z:z + 1, :]
            df = pd.concat(
                [df, pd.DataFrame({'X': [coords[0, 0]], 'Y': [coords[0, 1]], 'T': [coords[0, 2]], 'Z': [sgs[z]]})],
                sort=False)

        return sgs
######################################

# Other Functions

######################################


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


def aggregate(df, list_zz, grid_params, mean_agg=True, weight_col=None):
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
        if weight_col:
            list_cols.append(weight_col)

        # Convert DataFrame to NumPy array for processing
        np_data = df[list_cols].to_numpy().astype(float)
        origin = np.array([xmin, ymin, tmin])
        resolution = np.array([res_space, res_space, res_time])

        # Calculate grid indices for each data point
        grid_indices = np.rint(((np_data[:, :3] - origin) / resolution)).astype(int)

        # Initialize 3D aggregation arrays
        grid_shape = (len(x_range), len(y_range), len(t_range))
        grid_sum = np.zeros(grid_shape)
        grid_count = np.zeros(grid_shape)

        if weight_col:
            grid_weighted_sum = np.zeros(grid_shape)
            grid_weight_sum = np.zeros(grid_shape)

        # Aggregate data into the grid
        for i in range(np_data.shape[0]):
            x_idx, y_idx, t_idx = grid_indices[i, :3]
            if x_idx < grid_shape[0] and y_idx < grid_shape[1] and t_idx < grid_shape[2]:
                if weight_col:
                    weight = np_data[i, 4]  # Weight column index
                    grid_weighted_sum[x_idx, y_idx, t_idx] += np_data[i, 3] * weight
                    grid_weight_sum[x_idx, y_idx, t_idx] += weight
                else:
                    grid_sum[x_idx, y_idx, t_idx] += np_data[i, 3]
                    grid_count[x_idx, y_idx, t_idx] += 1

        # Compute weighted or unweighted mean
        if mean_agg:
            with np.errstate(invalid='ignore', divide='ignore'):
                if weight_col:
                    grid_matrix = np.divide(grid_weighted_sum, grid_weight_sum)
                else:
                    grid_matrix = np.divide(grid_sum, grid_count)
        else:
            grid_matrix = grid_weighted_sum if weight_col else grid_sum

        # Flatten and update DataFrame
        df_grid[zz] = grid_matrix.ravel()

    return df_grid

def plot_histograms(df_grid, zz, nzz, maxlag, sample, interp_method, keep_nanapis, threshold, data_source):
    # Set a larger figure size
    plt.figure(figsize=(12, 6))

    # Plot original C1 histogram
    plt.subplot(121)
    plt.hist(df_grid[zz], facecolor='red', bins=50, alpha=0.2, edgecolor='black')
    plt.xlabel(zz)
    plt.ylabel('Frequency')
    plt.title('Original')
    plt.grid(True)

    # Plot normal score C1 histogram (with weights)
    plt.subplot(122)
    plt.hist(df_grid[nzz], facecolor='red', bins=50, alpha=0.2, edgecolor='black')
    plt.xlabel(nzz)
    plt.ylabel('Frequency')
    plt.title('Normalized')
    plt.grid(True)

    # Adjust subplot spacing
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=0.3)

    # Define path for saving the figure
    figures_path = '/scratch/users/pburdeau/notebooks/figures_out/'
    file_name = (f'hist_data_source_{data_source}_maxlag_{maxlag}_comp_{zz}_'
                 f'keep_nanapis_{keep_nanapis}_interp_method_{interp_method}_'
                 f'sample_{sample}_threshold_{threshold}.pdf')

    # Save figure with tight layout to ensure nothing is cut off
    plt.savefig(figures_path + file_name, bbox_inches='tight')
    plt.close()  # Close the figure to avoid overlapping in subsequent plots


def plot_variogram(V1, comp, basin, maxlag, sample, interp_method, keep_nanapis, threshold, data_source):
    # extract variogram values
    xdata = V1.bins
    ydata = V1.experimental

    # use exponential variogram model
    V1.model = 'exponential'

    # use Gaussian model
    V2 = V1
    V2.model = 'gaussian'

    # use spherical model
    V3 = V1
    V3.model = 'spherical'

    # evaluate models
    xi = np.linspace(0, xdata[-1], 100)

    y_exp = [models.exponential(h, V1.parameters[0], V1.parameters[1], V1.parameters[2]) for h in xi]
    y_gauss = [models.gaussian(h, V2.parameters[0], V2.parameters[1], V2.parameters[2]) for h in xi]
    y_sph = [models.spherical(h, V3.parameters[0], V3.parameters[1], V3.parameters[2]) for h in xi]
    #     y_cub = [models.cubic(h, V4.parameters[0], V4.parameters[1], V4.parameters[2]) for h in xi]

    # plot variogram model
    fig = plt.figure()
    plt.plot(xdata / 1000, ydata, 'og', label="Experimental variogram")
    # plt.plot(xi / 1000, y_gauss, 'b--', label='Gaussian variogram')
    plt.plot(xi / 1000, y_exp, 'b-', label='Exponential variogram')
    # plt.plot(xi / 1000, y_sph, 'b*-', label='Spherical variogram')
    #     plt.plot(xi/1000, y_cub,'b*-', label='Cubic variogram')

    plt.title(f'Isotropic variogram for {comp} with maxlag {maxlag}')
    plt.xlabel('Lag [km]');
    plt.ylabel('Semivariance')
    #     plt.subplots_adjust(left=0.0, bottom=0.0, right=1, top=1.0) # adjust the plot size
    plt.legend(loc='lower right')
    # plt.show()
    variograms_path = root_path + '/out/variograms/'
    os.makedirs(variograms_path, exist_ok=True)
    figures_path = '/scratch/users/pburdeau/notebooks/figures_out/'

    # plt.savefig(
    #     figures_path + f'vario_data_source_{data_source}_basin_{basin}_comp_{comp}_keep_nanapis_{keep_nanapis}_interp_method_{interp_method}_sample_{sample}_threshold_{threshold}.pdf'
    # )

def create_variogram(data_single, comp, basin, n_lags, maxlag, sample, interp_method, keep_nanapis, threshold, data_source):
    # New method to include uncertainties
    data = data_single[comp].values.reshape(-1, 1)
    uncertainties = data_single['std_' + comp].values.reshape(-1, 1)

    # Apply QuantileTransformer
    nst_trans_comp = QuantileTransformer(n_quantiles=500, output_distribution='normal').fit(data)
    data_single['norm_' + comp] = nst_trans_comp.transform(data)
    data_single['norm_std_' + comp] = nst_trans_comp.transform(uncertainties)

    # Check transformed variability
    if len(np.unique(data_single['norm_' + comp])) <= 1 or np.std(data_single['norm_' + comp]) < 1e-6:
        print(f"Insufficient variability for {comp} in {basin}. Using average variogram.")

        # Use the average variogram
        output_dir = os.path.join(root_path, 'average_variograms',
                                  f'{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}')
        variogram_file = os.path.join(output_dir, f'{comp}_variogram.json')
        try:
            with open(variogram_file, 'r') as f:
                variogram = json.load(f)

                # Extract parameters
                azimuth = variogram.get("azimuth", 0)
                nugget = variogram.get("nugget", 0.1)
                major_range = variogram.get("major_range", 1.0)
                minor_range = variogram.get("minor_range", 1.0)
                sill = variogram.get("sill", 1.0)
                vtype = variogram.get("type", 'Exponential')
        except FileNotFoundError:
            print(f"Fallback variogram file not found for {comp} in {basin}. Using default parameters.")
            azimuth = 0
            nugget = 0.1  # Default fallback
            major_range = 1.0  # Default fallback
            minor_range = 1.0  # Default fallback
            sill = 1.0  # Default fallback
            vtype = 'Exponential'

        return nst_trans_comp, azimuth, nugget, major_range, minor_range, sill, vtype

    # Proceed with variogram creation if variability is sufficient
    coords = data_single[['X', 'Y', 'T']].values
    values = data_single['norm_' + comp]

    try:
        # Attempt to create the variogram
        V1 = skg.Variogram(coords, values, bin_func='even', n_lags=n_lags, maxlag=maxlag, normalize=False)
        plot_variogram(V1, comp, basin, maxlag, sample, interp_method, keep_nanapis, threshold, data_source)

        azimuth = 0
        nugget = V1.parameters[2]

        # isotropic
        major_range = V1.parameters[0]
        minor_range = V1.parameters[0]
        sill = V1.parameters[1]
        vtype = 'Exponential'
        print(f'Created variogram in {basin} for {comp}.')
    except RuntimeError as e:
        print(f"Variogram creation failed for {comp} in {basin}: {e}. Using average variogram.")
        # Fallback to average variogram parameters if creation fails
        output_dir = os.path.join(root_path, 'average_variograms',
                                  f'{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}')
        variogram_file = os.path.join(output_dir, f'{comp}_variogram.json')
        try:
            with open(variogram_file, 'r') as f:
                variogram = json.load(f)

                # Extract parameters
                azimuth = variogram.get("azimuth", 0)
                nugget = variogram.get("nugget", 0.1)
                major_range = variogram.get("major_range", 1.0)
                minor_range = variogram.get("minor_range", 1.0)
                sill = variogram.get("sill", 1.0)
                vtype = variogram.get("type", 'Exponential')
        except FileNotFoundError:
            print(f"Fallback variogram file not found for {comp} in {basin}. Using default parameters.")
            azimuth = 0
            nugget = 0.1  # Default fallback
            major_range = 1.0  # Default fallback
            minor_range = 1.0  # Default fallback
            sill = 1.0  # Default fallback
            vtype = 'Exponential'

    return nst_trans_comp, azimuth, nugget, major_range, minor_range, sill, vtype


def interpolate_comp(comp, grid, data_on_grid, basin, n_lags, maxlag, k, rad,
                     sample, interp_method, keep_nanapis, threshold, data_source):
    print('Start interpolating ' + comp + '...')
    data_single = data_on_grid[~pd.isna(data_on_grid[comp])].reset_index(drop=True)
    data_single = data_single[data_single[comp] > 0].reset_index(drop=True)
    print(f'{len(data_single)} initial datapoints for {comp} in {basin}')
    if len(data_single) > 0:

        if len(data_single) >= 50 and len(np.unique(data_single[comp])) > 1 and np.std(data_single[comp]) > 1e-6:
            print(f'Trying to create variogram in {basin} for {comp}')

            nst_trans_comp, azimuth, nugget, major_range, minor_range, sill, vtype = create_variogram(
                data_single, comp, basin, n_lags, maxlag, sample, interp_method, keep_nanapis, threshold, data_source)
        else:  # use average values computed before
            output_dir = os.path.join(root_path, 'average_variograms',
                                      f'{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}')
            variogram_file = os.path.join(output_dir, f'{comp}_variogram.json')
            with open(variogram_file, 'r') as f:
                variogram = json.load(f)

                # Extract parameters
                azimuth = variogram.get("azimuth")
                nugget = variogram.get("nugget")
                major_range = variogram.get("major_range")
                minor_range = variogram.get("minor_range")
                sill = variogram.get("sill")
                vtype = variogram.get("type")

            data = data_single[comp].values.reshape(-1, 1)
            uncertainties = data_single['std_' + comp].values.reshape(-1, 1)
            nst_trans_comp = QuantileTransformer(n_quantiles=500, output_distribution='normal').fit(data)
            data_single['norm_' + comp] = nst_trans_comp.transform(data)
            data_single['norm_std_' + comp] = nst_trans_comp.transform(uncertainties)

        vario = [azimuth, nugget, major_range, minor_range, sill, vtype]

        if interp_method.startswith('sgs') and interp_method in [f'sgs_{i}' for i in range(1, 11)]:
            # Call the Sequential Gaussian Simulation function
            data_out = Interpolation.okrige_sgs_3d(grid, data_single, 'X', 'Y', 'T', 'norm_' + comp, 'norm_std_' + comp,
                                                   k,
                                                   vario, rad)

            var_out = np.zeros(len(data_out))

            print('Done interpolating with SGS for ' + comp + ' in ' + basin + '...')

        elif interp_method == 'nn':
            print('Start interpolating with nearest neighbor ' + comp + 'in ' + basin + '...')

            data_out, var_out = Interpolation.nearest_neighbor_estimation(grid,
                                                                          data_single, 'X', 'Y', 'T', 'norm_' + comp,
                                                                          rad)
            print('Done interpolating with nearest neighbor ' + comp + 'in ' + basin + '...')

        elif interp_method == 'ordinary_kriging':
            data_out, var_out = Interpolation.okrige_3d(grid,
                                                        data_single, 'X', 'Y', 'T', 'norm_' + comp,
                                                        'norm_std_' + comp,
                                                        k, vario, rad)

        elif interp_method == 'nn_no_time':
            data_out, var_out = Interpolation.nearest_neighbor_estimation_no_time(grid,
                                                                          data_single, 'X', 'Y', 'T', 'norm_' + comp,
                                                                          rad)
        # okrige_3d(prediction_grid, df, xx, yy, tt, zz, uncertainty_col, num_points, vario, radius, quiet=False):

        # reverse normal score transformation
        var_out[var_out < 0] = 0;  # make sure variances are non-negative
        std_out = np.sqrt(var_out)  # convert to standard deviation (this should be done before back transforming!!!)

        # reshape
        data_out = data_out.reshape(-1, 1)
        std_out = std_out.reshape(-1, 1)
        # back transformation
        data_out = nst_trans_comp.inverse_transform(data_out)
        std_out = nst_trans_comp.inverse_transform(std_out)
        std_out = std_out - np.nanmin(std_out)

    else:
        data_out = [np.nan for i in range(len(grid))]
        std_out = [np.nan for i in range(len(grid))]

    # add to initial data
    data_on_grid = data_on_grid.copy()

    data_on_grid[comp] = data_out
    data_on_grid['std_' + comp] = std_out

    return data_on_grid


def interpolate_all_components(components, basin, df_data, df_prod, res_space, res_time,
                               alpha, k, rad, maxlag, n_lags, interp_method,
                               frac, sample, a, b, keep_nanapis, threshold, data_source, radius_factor):
    sorted_components = components
    grid, grid_params = create_grid(df_prod, res_space, res_time)
    prod_on_grid = aggregate(df_prod, ['Gas'], grid_params, mean_agg=False)
    data_on_grid = aggregate(df_data, components, grid_params, mean_agg=True)
    prod_oil_on_grid = aggregate(df_prod, ['Oil'], grid_params, mean_agg=False)
    data_on_grid = data_on_grid[prod_on_grid.Gas > 0].reset_index(drop=True)

    if a != 2 and data_on_grid[~pd.isna(data_on_grid.C1)][['X', 'Y']].drop_duplicates().shape[0] > 5:
        np.random.seed(42)

        # Parameters
        nb_clusters = 4  # Number of clusters
        # Extract X and Y coordinates

        data_on_grid_not_na = data_on_grid[~pd.isna(data_on_grid.C1)]

        coordinates = data_on_grid_not_na[['X', 'Y']].to_numpy()

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
        data_on_grid_not_na['Cluster'] = kmeans.fit_predict(coordinates)

        # Get cluster centers
        centers = kmeans.cluster_centers_

        # Define a fixed radius based on the dataset's spatial extent
        x_range = np.max(data_on_grid_not_na['X']) - np.min(data_on_grid_not_na['X'])
        y_range = np.max(data_on_grid_not_na['Y']) - np.min(data_on_grid_not_na['Y'])
        radius = radius_factor * min(x_range, y_range)

        # Compute distances between all cluster centers
        distances = cdist(centers, centers, metric='euclidean')

        # Track which centers have been used for test and validation sets
        used_for_selector_0 = set()
        used_for_selector_1 = set()

        # Generate exactly 4 pairs of selectors, along with their center indices
        selector_pairs = []

        while len(selector_pairs) < nb_clusters:
            # Select the next unused center for selector_0
            selector_0_idx = next(idx for idx in range(nb_clusters) if idx not in used_for_selector_0)

            # From remaining centers, select the farthest unused center for selector_1
            available_selector_1 = [idx for idx in range(nb_clusters) if idx not in used_for_selector_1 and idx != selector_0_idx]
            if not available_selector_1:
                # Add fallback logic to reuse already used selectors
                available_selector_1 = [idx for idx in range(nb_clusters) if idx != selector_0_idx]

            if not available_selector_1:
                raise ValueError(
                    f"No more unused centers available for validation sets in {basin} for threshold = {threshold}, keep_nanapis = {keep_nanapis}, sample = {sample}, (a, b) = ({a}, {b}) and radius_factor = {radius_factor}")

            farthest_idx = max(available_selector_1, key=lambda idx: distances[selector_0_idx, idx])

            # Mark the selected centers as used
            used_for_selector_0.add(selector_0_idx)
            used_for_selector_1.add(farthest_idx)

            # Create selectors
            distances_0 = np.sqrt((data_on_grid['X'] - centers[selector_0_idx][0]) ** 2 +
                                  (data_on_grid['Y'] - centers[selector_0_idx][1]) ** 2)
            selector_0 = distances_0 <= radius

            distances_1 = np.sqrt((data_on_grid['X'] - centers[farthest_idx][0]) ** 2 +
                                  (data_on_grid['Y'] - centers[farthest_idx][1]) ** 2)
            selector_1 = (distances_1 <= radius) & ~selector_0

            # Save the pair along with their center indices
            selector_pairs.append(((selector_0, selector_0_idx), (selector_1, farthest_idx)))

        def map_to_index(a_list, b_list):
            pairs = list(zip(a_list, b_list))
            return {pair: idx for idx, pair in enumerate(pairs)}

        i = map_to_index([-1, 1, -1, 1], [-1, 1, 1, -1])[(a, b)]
        selector_0, selector_0_center_idx = selector_pairs[i][0]
        selector_1, selector_1_center_idx = selector_pairs[i][1]

        selector = selector_0 | selector_1

        data_on_grid_not_selected_0 = data_on_grid[selector_0]
        data_on_grid_not_selected_0['BASIN_NAME'] = [basin for i in range(len(data_on_grid_not_selected_0))]

        data_on_grid_not_selected_0.to_csv(os.path.join(root_path, 'out',
                                                        f'block_validation_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_sample_{sample}_a_{a}_b_{b}_basin_{basin}_radius_factor_{radius_factor}.csv'))

        data_on_grid_not_selected_1 = data_on_grid[selector_1]
        data_on_grid_not_selected_1['BASIN_NAME'] = [basin for i in range(len(data_on_grid_not_selected_1))]

        data_on_grid_not_selected_1.to_csv(os.path.join(root_path, 'out',
                                                        f'block_test_{data_source}_threshold_{threshold}_keep_nanapis_{keep_nanapis}_sample_{sample}_a_{a}_b_{b}_basin_{basin}_radius_factor_{radius_factor}.csv'))

        # Plot only selector_0 with a radius circle (without the legend)
        plt.figure(figsize=(16, 12))
        ax = plt.gca()  # Get the current axis

        # Add the radius circle for selector_0
        circle = Circle(centers[selector_0_center_idx], radius, color='#93a09d', alpha=0.12, label='Holdout area')
        ax.add_patch(circle)

        # Plot the input set
        data_on_grid_not_na = data_on_grid[~pd.isna(data_on_grid.C1)]
        plt.scatter(data_on_grid_not_na['X'], data_on_grid_not_na['Y'], color='#d2c9b6', s=42, label='Input set')

        # Plot data points for selector_0
        data_on_grid_not_selected_0_not_na = data_on_grid[selector_0 & ~pd.isna(data_on_grid.C1)]
        plt.scatter(data_on_grid_not_selected_0_not_na['X'], data_on_grid_not_selected_0_not_na['Y'],
                    color='#93a09d', s=42, label='Holdout set')

        # Plot the cluster center
        plt.scatter(centers[selector_0_center_idx, 0], centers[selector_0_center_idx, 1], color='black', s=200, marker='X',
                    label='Cluster center')

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Finalize the plot without the legend
        plt.axis('off')
        plt.tight_layout()
        figures_path = '/scratch/users/pburdeau/notebooks/figures_out/'

        # Save the figure without the legend
        plt.savefig(
            figures_path + f'plot_test_block_no_legend_{data_source}_basin_{basin}_keep_nanapis_{keep_nanapis}_sample_{sample}_threshold_{threshold}_radius_factor_{radius_factor}_a_{a}_b_{b}.png'
        )
        plt.show()

        # Create a separate figure for the legend only
        fig_legend = plt.figure(figsize=(12, 9))

        # Custom legend with individual marker scales
        legend_handles = [
            Line2D([0], [0], marker='o', color='w', label='Input set', markerfacecolor='#d2c9b6', markersize=20),
            Line2D([0], [0], marker='o', color='w', label='Holdout set', markerfacecolor='#93a09d', markersize=20),
            Line2D([0], [0], marker='X', color='w', label='Cluster center', markerfacecolor='black', markersize=20),
            Patch(facecolor='#93a09d', edgecolor='none', alpha=0.12, label='Holdout area')
        ]

        # Add the legend to the separate figure
        fig_legend.legend(handles=legend_handles, loc='center', fontsize=24, frameon=True)
        plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)  # Adjust layout for the legend-only figure
        plt.axis('off')  # Turn off the axis for a clean legend-only figure

        # Save the legend-only figure
        plt.savefig(
            figures_path + f'legend_plot_test_block_{data_source}_keep_nanapis_{keep_nanapis}_sample_{sample}_threshold_{threshold}_radius_factor_{radius_factor}_a_{a}_b_{b}.png'
        )
        plt.show()

        # Plot the main figure
        plt.figure(figsize=(16, 12))

        # Plot the input set
        plt.scatter(data_on_grid_not_na['X'], data_on_grid_not_na['Y'], color='#d2c9b6', s=24, label='Input sets')

        # Add the radius circles for each cluster
        ax = plt.gca()  # Get the current axis
        for idx, center in enumerate(centers):
            circle = Circle(center, radius, color='#93a09d', alpha=0.07, label='Holdout areas' if idx == 0 else None)
            ax.add_patch(circle)

        j = 0
        # Plot the selector pairs

        for (selector_0, selector_0_center_idx), (selector_1, selector_1_center_idx) in selector_pairs:
            # First selector
            data_on_grid_selected_0 = data_on_grid[selector_0]
            data_on_grid_not_selected_0_not_na = data_on_grid_selected_0[~pd.isna(data_on_grid_selected_0.C1)]
            plt.scatter(data_on_grid_not_selected_0_not_na['X'], data_on_grid_not_selected_0_not_na['Y'],
                        s=24, color='#93a09d')

            # Second selector
            data_on_grid_selected_1 = data_on_grid[selector_1]
            data_on_grid_not_selected_1_not_na = data_on_grid_selected_1[~pd.isna(data_on_grid_selected_1.C1)]
            plt.scatter(data_on_grid_not_selected_1_not_na['X'], data_on_grid_not_selected_1_not_na['Y'],
                        s=24, color='#93a09d')
            if j == 3:
                plt.scatter(data_on_grid_not_selected_1_not_na['X'], data_on_grid_not_selected_1_not_na['Y'],
                            s=24, color='#93a09d', label='Holdout sets')
            j += 1

        ax.set_aspect('equal', adjustable='box')

        # Plot cluster centers
        plt.scatter(centers[:, 0], centers[:, 1], color='black', s=200, marker='X', label='Cluster centers')

        # Add the legend
        plt.legend(loc='upper right', fontsize=24, markerscale=1.5, frameon=True)

        # Finalize the plot
        plt.axis('off')
        plt.tight_layout()

        plt.show()

        #     plt.show()

        selector_0, selector_0_center_idx = selector_pairs[i][0]
        selector_1, selector_1_center_idx = selector_pairs[i][1]
        selector = selector_0 | selector_1

        if sample == 1:  # if sample = 1, we only remove the block test
            selector = selector_1  # remove only test for total

        if not sample == 2:  # if sample = 2, we do not remove any block

            # For the data within the disks, replace the components with NaN
            data_on_grid.loc[selector, components] = np.nan  # replace with NaNs in kept

    df_grid = pd.DataFrame()
    df_grid['X'] = grid[:, 0]
    df_grid['Y'] = grid[:, 1]
    df_grid['T'] = grid[:, 2]

    prod_oil_on_grid = prod_oil_on_grid[prod_on_grid.Gas > 0].reset_index(drop=True)
    df_grid = df_grid[prod_on_grid.Gas > 0].reset_index(drop=True)
    grid = df_grid.values
    prod_on_grid = prod_on_grid[prod_on_grid.Gas > 0].reset_index(drop=True)
    for comp in components:
        data_on_grid['std_' + comp] = 0  # no initial uncertainties assumption for USGS

    i = 0
    for comp in components:
        i += 1

        data_on_grid = interpolate_comp(comp, grid, data_on_grid,
                                        basin, n_lags,
                                        maxlag, k, rad, sample, interp_method, keep_nanapis, threshold, data_source)

        print(f'Component {i} on {len(sorted_components)} done.')
    data_on_grid['Gas'] = prod_on_grid['Gas']
    data_on_grid['Oil'] = prod_oil_on_grid['Oil']

    # Do not normalize now, do the analysis on notebook

    data_on_grid['check_sum'] = data_on_grid[components].sum(axis=1)

    return data_on_grid


def my_function(data_source, basin, interp_method, sample, keep_nanapis, threshold, radius_factor):
    ################################################################################################################################

    # download data

    ################################################################################################################################
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
        data = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'))  # all basins

    ################################################################################################################################

    # download wells data

    ################################################################################################################################

    wells_data = pd.read_csv(os.path.join(root_path, f'wells_info_prod_per_basin/{basin}_final.csv'))  # only basin data

    ################################################################################################################################

    # interpolate

    ################################################################################################################################

    # define parameters
    frac = 0.9
    alpha = 6.34

    n_lags = 10
    k = 100  # number of neighboring data points used to estimate a given point

    res_space = 2000 / alpha # replace by 200? (almost by wellsite?)
    res_time = 1825  # 5 years # replace by 365 (By year)
    maxlag = 100000 / alpha
    rad = maxlag

    df_data = data.reset_index(drop=True)
    df_data = df_data[df_data.BASIN_NAME == basin].reset_index(drop=True)
    df_data['X'] = df_data.X / alpha
    df_data['Y'] = df_data.Y / alpha

    df_prod = wells_data[wells_data.BASIN_NAME == basin].reset_index(drop=True)
    df_prod = df_prod.rename(columns={'Monthly Oil': 'Oil', 'Monthly Gas': 'Gas'})
    df_prod = df_prod[df_prod.Gas > 0].reset_index(drop=True)
    df_prod['X'] = df_prod.X / alpha
    df_prod['Y'] = df_prod.Y / alpha

    df_prod = df_prod[~pd.isna(df_prod['Gas'])].reset_index(drop=True)

    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2', 'C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
    components = [comp for comp in components if comp in df_data.columns]

    if interp_method.startswith('sgs') and interp_method in [f'sgs_{i}' for i in range(1, 11)]:
        a = 2
        b = 2
        data_on_grid = interpolate_all_components(components, basin, df_data, df_prod,
                                                  res_space, res_time, alpha, k, rad, maxlag, n_lags,
                                                  interp_method, frac, sample, a, b, keep_nanapis, threshold,
                                                  data_source, radius_factor)

        data_on_grid['BASIN_NAME'] = basin
        data_on_grid.to_csv(os.path.join(root_path, 'out',
                                         f'test_result_data_source_{data_source}_a_{a}_b_{b}_basin_{basin}_sample_{sample}_interp_{interp_method}_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv'))
    else:

        for a, b in zip([-1, 1, -1, 1], [-1, 1, 1, -1]):
            data_on_grid = interpolate_all_components(components, basin, df_data, df_prod,
                                                      res_space, res_time, alpha, k, rad, maxlag, n_lags,
                                                      interp_method, frac, sample, a, b, keep_nanapis, threshold,
                                                      data_source, radius_factor)
            data_on_grid['BASIN_NAME'] = basin
            data_on_grid.to_csv(os.path.join(root_path, 'out',
                                             f'result_data_source_{data_source}_a_{a}_b_{b}_basin_{basin}_sample_{sample}_interp_{interp_method}_keep_nanapis_{keep_nanapis}_threshold_{threshold}_radius_factor_{radius_factor}.csv'))

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
        'Gulf Coast Basin (LA, TX)',
        'San Joaquin Basin',
        'Powder River Basin',
        'Chautauqua Platform',
        'Arkoma Basin',
        'San Juan Basin',
        'Green River Basin',
        'Paradox Basin',
        'Piceance Basin',
        'Appalachian Basin (Eastern Overthrust Area)',
        'Palo Duro Basin',
        'East Texas Basin',
        'Denver Basin',
        'Arkla Basin',
        'South Oklahoma Folded Belt',
        'Bend Arch',
        'Permian Basin',
        'Fort Worth Syncline',
        'Wind River Basin',
        'Uinta Basin',
        'Williston Basin',
        'Las Animas Arch',
        'Appalachian Basin',
        'Anadarko Basin'
    ]

    # basin = basins_many_datapoints[arguments.basin_id]

    bools = [0, 1, 2]
    sample = bools[arguments.sample_id]
    # 'nn_no_time'
    interp_methods = ['nn',
                      'ordinary_kriging',
                      'sgs_1',
                      'sgs_2',
                      'sgs_3',
                      'sgs_4',
                      'sgs_5']
    interp_method = interp_methods[arguments.interp_method_id]

    data_sources = ['usgs', 'ghgrp']
    data_source = data_sources[arguments.data_source]

    thresholds = [1000, 25, 50, 10]
    threshold = thresholds[arguments.threshold]

    keep_nanapiss = [True, False]
    keep_nanapis = keep_nanapiss[arguments.keep_nanapis]

    radius_factors = [0.15, 0.10]
    radius_factor = radius_factors[arguments.radius_factor]

    # Get the number of CPUs allocated by SLURM
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", 16))

    # Divide basins into groups based on the available CPUs
    basin_groups = np.array_split(basins_many_datapoints, num_cpus)

    # Process each group in parallel
    Parallel(n_jobs=num_cpus)(
        delayed(my_function)(data_source, basin, interp_method, sample, keep_nanapis, threshold, radius_factor)
        for group in basin_groups for basin in group
    )
