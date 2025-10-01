#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: Enverus Anadarko — Clean, geotag, merge, and organize well data by basin.

STEPS
1) Select and clean “useful columns” from raw production and header CSVs.
2) Convert headers to GeoDataFrame and spatially join to basin polygons.
3) Merge production with geotagged headers (by API14).
4) Split merged files by BASIN_NAME.
5) Concatenate basin-split chunks into one file per basin.

"""

# Optional import per your environment preference (no-ops if module not present)
# from di_scrubbing_func import *  # <- uncomment if you want this available by default

import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm


# ------------------------------ #
#           PATHS/IO             #
# ------------------------------ #

# Root folders (adjust if needed)
base_folder = '/content/drive/Shared drives/Enverus US'
output_folder = os.path.join(base_folder, 'Processed Data')

# Raw inputs (as delivered)
raw_wells_production_folder = os.path.join(base_folder, 'Well Monthly Production - Anadarko')
raw_wells_headers_folder    = os.path.join(base_folder, 'Well Headers - Anadarko')

# Stage 1 outputs (column-filtered raw files)
wells_production_folder = os.path.join(output_folder, 'Well Monthly Production with useful columns - Anadarko')
wells_headers_folder    = os.path.join(output_folder, 'Well Headers with useful columns - Anadarko')

# Stage 2–5 outputs
headers_output_folder = os.path.join(output_folder, 'Well Headers with Basins')
merged_output_folder  = os.path.join(output_folder, 'Well Monthly Production with Headers and Basins')
split_output_folder   = os.path.join(output_folder, 'Well Monthly Production with Headers and Basins split by Basin')
final_output_folder   = os.path.join(output_folder, 'Well Monthly Production with Headers and Basins concatenated by Basin')

# Create all output dirs
for _p in [
    output_folder, wells_production_folder, wells_headers_folder,
    headers_output_folder, merged_output_folder, split_output_folder, final_output_folder
]:
    os.makedirs(_p, exist_ok=True)

# Basins shapefile (folder or file path acceptable to GeoPandas)
basin_gdf_path = os.path.join(base_folder, 'Basins_Shapefile')


# ------------------------------ #
#        STAGE 1: CLEAN RAW      #
# ------------------------------ #

# 1A) Production: keep only core columns and filter invalid API/UWI
production_csv_files = [f for f in os.listdir(raw_wells_production_folder)]
for file in tqdm(production_csv_files, desc="Stage 1A — Clean production CSVs"):
    file_path = os.path.join(raw_wells_production_folder, file)
    # Read only needed columns for speed/memory
    processed_df = pd.read_csv(
        file_path,
        usecols=['API/UWI', 'Monthly Oil', 'Monthly Gas', 'Monthly Production Date']
    )

    # Drop missing/zero API
    processed_df = processed_df[~pd.isna(processed_df['API/UWI'])].copy()
    processed_df['API/UWI'] = processed_df['API/UWI'].astype(str)
    processed_df = processed_df[processed_df['API/UWI'] != '0']

    # Save cleaned file
    output_path = os.path.join(wells_production_folder, file)
    processed_df.to_csv(output_path, index=False)

# 1B) Headers: keep only core columns and filter invalid API14
headers_csv_files = [f for f in os.listdir(raw_wells_headers_folder) if f.lower().endswith('.csv')]
for file in tqdm(headers_csv_files, desc="Stage 1B — Clean header CSVs"):
    file_path = os.path.join(raw_wells_headers_folder, file)
    processed_df = pd.read_csv(
        file_path,
        usecols=['API14', 'Surface Hole Latitude (WGS84)', 'Surface Hole Longitude (WGS84)']
    )

    # Drop missing/zero API14
    processed_df = processed_df[~pd.isna(processed_df['API14'])].copy()
    processed_df['API14'] = processed_df['API14'].astype(str)
    processed_df = processed_df[processed_df['API14'] != '0']

    # Save cleaned file
    output_path = os.path.join(wells_headers_folder, file)
    processed_df.to_csv(output_path, index=False)


# ------------------------------ #
#     STAGE 2: GEO + BASINS      #
# ------------------------------ #

def process_well_headers(headers_folder: str, basin_gdf: gpd.GeoDataFrame, output_folder: str) -> str:
    """
    Read cleaned header CSVs, convert to GeoDataFrame (EPSG:4326),
    reproject to EPSG:26914, and spatially join with basins (EPSG:26914).

    Parameters
    ----------
    headers_folder : str
        Folder containing cleaned header CSVs with API14 and WGS84 coordinates.
    basin_gdf : GeoDataFrame
        Basin polygons (any CRS; reprojected internally to 26914).
    output_folder : str
        Folder to write the joined CSV (API14, lat, lon, BASIN_NAME).

    Returns
    -------
    str
        Path to the combined header file with BASIN_NAME.
    """
    output_file = os.path.join(output_folder, 'wellheaderswithbasins.csv')

    # Read all headers into one DataFrame
    headers_files = [f for f in os.listdir(headers_folder) if f.lower().endswith('.csv')]
    headers_df = pd.concat(
        [pd.read_csv(os.path.join(headers_folder, file)) for file in tqdm(headers_files, desc="Stage 2 — Read headers")],
        ignore_index=True
    )

    # Build GeoDataFrame from lon/lat (WGS84)
    headers_gdf = gpd.GeoDataFrame(
        headers_df,
        geometry=[Point(xy) for xy in zip(headers_df['Surface Hole Longitude (WGS84)'],
                                          headers_df['Surface Hole Latitude (WGS84)'])],
        crs='EPSG:4326'
    )

    # Spatial join in projected CRS (EPSG:26914) for robust 'within' predicate
    headers_gdf_26914 = headers_gdf.to_crs(26914)
    basins_26914 = basin_gdf.to_crs(26914)

    # Keep only points strictly within a basin polygon
    joined = gpd.sjoin(headers_gdf_26914, basins_26914, how='inner', predicate='within')

    # Write minimal columns needed downstream
    joined[['API14', 'Surface Hole Latitude (WGS84)', 'Surface Hole Longitude (WGS84)', 'BASIN_NAME']].to_csv(
        output_file, index=False
    )
    return output_file


# ------------------------------ #
#     STAGE 3: MERGE PROD+HEAD   #
# ------------------------------ #

def process_production_files(production_folder: str, headers_file: str, output_folder: str) -> None:
    """
    Merge each cleaned production CSV with the geotagged headers (by API).

    Parameters
    ----------
    production_folder : str
        Folder with cleaned production CSVs (must include 'API/UWI').
    headers_file : str
        CSV produced by `process_well_headers` containing API14 and BASIN_NAME.
    output_folder : str
        Destination folder for merged CSVs (one per input file).
    """
    headers_df = pd.read_csv(headers_file)
    headers_df['API14'] = headers_df['API14'].astype(str)

    production_files = [f for f in os.listdir(production_folder) if f.lower().endswith('.csv')]
    for file in tqdm(production_files, desc="Stage 3 — Merge production+headers"):
        production_df = pd.read_csv(os.path.join(production_folder, file))
        production_df['API/UWI'] = production_df['API/UWI'].astype(str)

        # Left-join to retain all production rows; BASIN_NAME may be NaN if no match
        merged_df = pd.merge(
            production_df,
            headers_df,
            left_on='API/UWI',
            right_on='API14',
            how='left'
        )

        # Persist merged file
        output_file = os.path.join(output_folder, f"processed_{file}")
        merged_df.to_csv(output_file, index=False)


# ------------------------------ #
#     STAGE 4: SPLIT BY BASIN    #
# ------------------------------ #

def split_by_basin(input_folder: str, output_folder: str) -> None:
    """
    Split each merged file into multiple files (one per BASIN_NAME present).

    Parameters
    ----------
    input_folder : str
        Folder containing merged production+headers CSVs.
    output_folder : str
        Destination folder for basin-split CSVs.
    """
    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    for file in tqdm(input_files, desc="Stage 4 — Split by BASIN_NAME"):
        df = pd.read_csv(os.path.join(input_folder, file))

        # Skip files with no BASIN_NAME column (e.g., no spatial match)
        if 'BASIN_NAME' not in df.columns:
            continue

        # Group by basin and write each group
        for basin_name, group in df.groupby('BASIN_NAME'):
            # Defensive: handle NaN basin names
            if pd.isna(basin_name):
                basin_key = 'BASIN_UNKNOWN'
            else:
                basin_key = str(basin_name).replace(' ', '_')

            basin_filename = f"{basin_key}_{file}"
            basin_output_path = os.path.join(output_folder, basin_filename)
            group.to_csv(basin_output_path, index=False)


# ------------------------------ #
#   STAGE 5: CONCAT BY BASIN     #
# ------------------------------ #

def concat_by_basin(input_folder: str,
                    final_output_folder: str,
                    basins_gdf: gpd.GeoDataFrame,
                    processed_files=None):
    """
    Concatenate all split chunks for each basin into a single CSV per basin.

    Matching logic mirrors your original:
    - We look for files whose names start with the basin's name (spaces -> underscores),
      **and** that include the original 'processed_' tail (created in Stage 3).

    Parameters
    ----------
    input_folder : str
        Folder with basin-split CSVs from Stage 4.
    final_output_folder : str
        Where to write final per-basin CSVs.
    basins_gdf : GeoDataFrame
        Basin polygons (used only to enumerate basin names).
    processed_files : list[str] or None
        Optional list to track files already used (returned for chaining).

    Returns
    -------
    list[str]
        Updated list of processed file names.
    """
    if processed_files is None:
        processed_files = []

    for basin_name in tqdm(basins_gdf.BASIN_NAME.unique(), desc="Stage 5 — Concatenate by basin"):
        formatted_name = str(basin_name).replace(' ', '_')

        # Find split files belonging to this basin
        matching_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith('.csv') and f.split('_processed')[0] == formatted_name
        ]
        # Remove any already-processed files (idempotent runs)
        matching_files = [f for f in matching_files if f not in processed_files]

        if not matching_files:
            continue

        # Concatenate and stamp the canonical BASIN_NAME
        dfs = [pd.read_csv(os.path.join(input_folder, f)) for f in matching_files]
        processed_files.extend(matching_files)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df['BASIN_NAME'] = basin_name

        final_file = os.path.join(final_output_folder, f"{basin_name}_final.csv")
        final_df.to_csv(final_file, index=False)

    return processed_files


# ------------------------------ #
#             RUN IT             #
# ------------------------------ #

if __name__ == "__main__":
    # Load basins (any CRS; will be reprojected internally in Stage 2)
    basin_gdf = gpd.read_file(basin_gdf_path)

    # Stage 2: Spatially tag headers with basin names
    headers_file = process_well_headers(
        headers_folder=wells_headers_folder,
        basin_gdf=basin_gdf,
        output_folder=headers_output_folder
    )

    # Stage 3: Merge production records with geotagged headers
    process_production_files(
        production_folder=wells_production_folder,
        headers_file=headers_file,
        output_folder=merged_output_folder
    )

    # Stage 4: Split merged files by basin
    split_by_basin(
        input_folder=merged_output_folder,
        output_folder=split_output_folder
    )

    # Stage 5: Concatenate per-basin chunks
    concat_by_basin(
        input_folder=split_output_folder,
        final_output_folder=final_output_folder,
        basins_gdf=basin_gdf
    )
