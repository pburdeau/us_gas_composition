#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline: USGS + GHGRP processing + production enrichment

What this script does (high level)
----------------------------------
1) Load shapefiles for US states, counties, and basins.
2) Clean & geotag USGS composition samples; aggregate to (X, Y, Year, date, T, BASIN_NAME).
3) (Optional) Correlation analysis: component fractions vs log(GOR) with simple OLS lines.
4) Build production-by-API/year dataset with X, Y, Year, T for Anadarko files.
5) Process GHGRP "onshore production" data and join to production + GHGRP well mappings.
6) Process GHGRP processing facilities; map each capacity point to nearest facility and annotate.

Notes
-----
- CRS conventions:
    * Input lon/lat are EPSG:4326; spatial work is done in EPSG:26914 (US feet/meters projection).
- GeoPandas:
    * Uses `predicate='within'` (modern) instead of deprecated `op='within'`.
- Paths:
    * Set ROOT/SHAPEFILE/USGS/GHGRP paths in the CONFIG section below.
- Minor fix:
    * `state_abbrev_to_name` → `state_abbreviations_to_names`
"""

# ------------------------------ #
#             IMPORTS            #
# ------------------------------ #

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from shapely.geometry import Point
from tqdm import tqdm
from scipy.stats import linregress

from shapely.ops import nearest_points
from shapely.validation import explain_validity


# ------------------------------ #
#            CONFIG              #
# ------------------------------ #

# >>>>> EDIT THESE PATHS FOR YOUR ENVIRONMENT <<<<<
root_path        = '/path/to/project/root'                     # e.g., '/content/drive/MyDrive/project'
shapefiles_path  = os.path.join(root_path, 'shapefiles')       # contains US states/counties & basins
usgs_path        = os.path.join(root_path, 'usgs', 'usgs.csv') # USGS raw CSV
ghgrp_path       = os.path.join(root_path, 'ghgrp', 'ghgrp.csv')

# Output folders
fig_out_dir      = os.path.join(root_path, 'figures_out')
os.makedirs(fig_out_dir, exist_ok=True)

# CRS constants
CRS_WGS84  = "EPSG:4326"
CRS_LOCAL  = 26914


# ------------------------------ #
#         LOADING SHAPES         #
# ------------------------------ #

def load_shapefiles():
    """
    Load US states, counties, and basin shapefiles.
    - Counties are renamed to "COUNTY (STATE)" and we also store COUNTY_NAME, STATE_NAME.
    - Counties with unmatched STATEFP are dropped.

    Returns
    -------
    (GeoDataFrame, GeoDataFrame, GeoDataFrame)
        us_states, us_counties (normalized), basins
    """
    states_fp   = os.path.join(shapefiles_path, 'cb_2018_us_state_20m', 'cb_2018_us_state_20m.shp')
    counties_fp = os.path.join(shapefiles_path, 'cb_2018_us_county_500k', 'cb_2018_us_county_500k.shp')
    basins_fp   = os.path.join(shapefiles_path, 'basins_shapefiles')

    us_states  = gpd.read_file(states_fp)
    us_counties = gpd.read_file(counties_fp)
    basins     = gpd.read_file(basins_fp)

    index_to_deletes = []
    for i in range(len(us_counties)):
        county_name = us_counties['NAME'].iloc[i].upper()
        state_fp    = us_counties['STATEFP'].iloc[i]
        # Match county's state by STATEFP and fetch its name
        state_match = us_states[us_states.STATEFP == state_fp]
        if len(state_match) > 0:
            state_name = state_match.reset_index(drop=True).NAME.iloc[0].upper()
            us_counties.at[i, 'NAME']        = f"{county_name} ({state_name})"
            us_counties.at[i, 'COUNTY_NAME'] = county_name
            us_counties.at[i, 'STATE_NAME']  = state_name
        else:
            index_to_deletes.append(i)

    us_counties = us_counties.drop(index=index_to_deletes)
    return us_states, us_counties, basins


# ------------------------------ #
#        USGS DATA CLEANING      #
# ------------------------------ #

def clean_usgs_data(basins):
    """
    Clean and geotag USGS gas composition samples, compute time index T,
    spatially join to basins, and export aggregated means per (X, Y, Year, date, T, BASIN_NAME).

    Parameters
    ----------
    basins : GeoDataFrame
        Basin polygons.

    Returns
    -------
    GeoDataFrame
        USGS points in CRS_LOCAL with composition columns and basin labels.
    """
    usgs = pd.read_csv(usgs_path)

    # Keep rows with valid coordinates and dates
    usgs = usgs[usgs.LAT.notna() & usgs.LONG.notna() & usgs.FINAL_SAMPLING_DATE.notna()].copy()
    usgs['Year'] = usgs['FINAL_SAMPLING_DATE'].str[-4:].astype(int)

    # To projected CRS for spatial ops
    usgs_gdf = gpd.GeoDataFrame(usgs,
                                geometry=gpd.points_from_xy(usgs.LONG, usgs.LAT),
                                crs=CRS_WGS84).to_crs(CRS_LOCAL)

    usgs_gdf = usgs_gdf.rename(columns={'API': 'API14'})
    usgs_gdf['COUNTY'] = usgs_gdf['COUNTY'] + ' (' + usgs_gdf['STATE'] + ')'

    # Parse dates robustly, clipping obvious out-of-range days
    dates = []
    for raw_date in tqdm(usgs_gdf['FINAL_SAMPLING_DATE'], desc="USGS: parse dates"):
        month, day, year = [int(x) for x in raw_date.split('/')]
        month = max(1, min(month, 12))
        # keep day in simple range (no month-specific day calcs, but avoids 31st bugs)
        day   = max(1, min(day, 30 if month != 2 else 28))
        dates.append(datetime(year, month, day))
    usgs_gdf['date'] = dates

    # Time index relative to first sample (days)
    usgs_gdf = usgs_gdf.sort_values('date').reset_index(drop=True)
    usgs_gdf['epochs'] = (usgs_gdf['date'] - usgs_gdf['date'].iloc[0]).dt.days

    # Spatial join to basins
    usgs_gdf = gpd.sjoin(usgs_gdf, basins.to_crs(CRS_LOCAL), how='inner', predicate='within')

    # Convenience columns
    usgs_gdf['X'] = usgs_gdf.geometry.x
    usgs_gdf['Y'] = usgs_gdf.geometry.y
    usgs_gdf['T'] = usgs_gdf['epochs']

    # Drop unneeded columns if present
    drop_cols = [
        'ID','SOURCE','BLM ID','USGS ID','ALASKA_QUAD_NAMES','WELL NAME','PUBLICATION DATE',
        'COUNTY','STATE','FINAL_DRILLING_DATE','FINAL_SAMPLING_DATE','SAMPLE DATE BEFORE COMPLETION',
        'index_right','BASIN_CODE','DEPTH','AGE','FORMATION','geometry','FIELD','LAT','LONG'
    ]
    usgs_gdf = usgs_gdf.drop(columns=[c for c in drop_cols if c in usgs_gdf.columns], errors='ignore')

    # Ensure numeric components
    components = ['HE','CO2','H2','N2','H2S','AR','O2','C1','C2','C3','N-C4','I-C4','N-C5','I-C5','C6+']
    usgs_gdf[components] = usgs_gdf[components].apply(pd.to_numeric, errors='coerce')

    # Recreate geometry after dropping column (stay in local CRS)
    usgs_gdf['geometry'] = gpd.points_from_xy(usgs_gdf['X'], usgs_gdf['Y'])
    usgs_gdf = gpd.GeoDataFrame(usgs_gdf, geometry='geometry', crs=CRS_LOCAL)

    # Aggregate means at (X, Y, Year, date, T, BASIN_NAME)
    key_cols = ['X','Y','Year','date','T','BASIN_NAME']
    agg_df = usgs_gdf.groupby(key_cols).agg({col: np.nanmean for col in components}).reset_index()

    # Persist
    out_csv = os.path.join(root_path, 'usgs', 'usgs_processed_with_nanapis.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    agg_df.to_csv(out_csv, index=False)

    return usgs_gdf


# ------------------------------ #
#       CORRELATION PLOTS        #
# ------------------------------ #

def correlation_analysis(usgs_gdf):
    """
    Scatter + OLS trend for selected components vs log(GOR).

    Expects columns: Monthly Gas, Monthly Oil, and component columns.
    (If Monthly Gas/Oil are not present in usgs_gdf, this will no-op gracefully.)
    """
    if not {'Monthly Gas', 'Monthly Oil'}.issubset(usgs_gdf.columns):
        print("Correlation analysis skipped: Monthly Gas/Oil not present in USGS frame.")
        return

    df = usgs_gdf.copy()
    df["GOR"] = df["Monthly Gas"] / df["Monthly Oil"]
    df["Log GOR"] = np.log(df["GOR"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["Log GOR"], inplace=True)

    components = ['C1','C2','C3','N-C4','I-C4','N-C5','I-C5','C6+']
    palette = sns.color_palette("crest", n_colors=len(components))[::-1]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i, (comp, color) in enumerate(zip(components, palette)):
        ax = axes[i]
        mask = df["Log GOR"].notna() & df[comp].notna()
        if mask.sum() > 1:
            ax.scatter(df.loc[mask, "Log GOR"], df.loc[mask, comp],
                       alpha=0.3, color=color, edgecolors="none")
            slope, intercept, r_value, _, _ = linregress(df.loc[mask, "Log GOR"], df.loc[mask, comp])
            x_vals = np.linspace(df["Log GOR"].min(), df["Log GOR"].max(), 100)
            ax.plot(x_vals, slope * x_vals + intercept, color="black")
            ax.text(0.05, 0.9, f"r={r_value:.3f}", transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.5))
        ax.set_title(comp, fontsize=10, fontweight="bold")
        ax.set_xlabel("Log GOR")
        ax.set_ylabel(f"% {comp}")

    plt.tight_layout()
    out_png = os.path.join(fig_out_dir, 'correlation_plots.png')
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved correlation plots to: {out_png}")


# ------------------------------ #
#     PRODUCTION (ADD YEAR/T)    #
# ------------------------------ #

def add_T_to_production_files():
    """
    Read Anadarko-labeled monthly production CSVs, compute projected X/Y,
    extract Year from 'Monthly Production Date', and compute T as days since 2000-01-01.
    Writes enriched CSVs to 'wells_info_prod_per_basin'.
    """
    input_folder  = os.path.join(root_path, 'wells_prod_jan_26')
    output_folder = os.path.join(root_path, 'wells_info_prod_per_basin')
    os.makedirs(output_folder, exist_ok=True)

    for file in tqdm(os.listdir(input_folder), desc="Prod: add X/Y/Year/T"):
        if file.endswith('.csv') and 'Anadarko' in file:
            df = pd.read_csv(os.path.join(input_folder, file))

            # Build geometry from lon/lat in WGS84, then project to local
            df['geometry'] = df.apply(
                lambda row: Point(row['Surface Hole Longitude (WGS84)'], row['Surface Hole Latitude (WGS84)']),
                axis=1
            )
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=CRS_WGS84).to_crs(CRS_LOCAL)
            gdf['X'], gdf['Y'] = gdf.geometry.x, gdf.geometry.y

            # Time fields
            gdf['Monthly Production Date'] = pd.to_datetime(gdf['Monthly Production Date'])
            gdf['Year'] = gdf['Monthly Production Date'].dt.year

            grouped = gdf.groupby(['API14','X','Y','Year'], as_index=False).agg({
                'Monthly Gas': 'sum',
                'Monthly Oil': 'sum',
                'BASIN_NAME' : 'first'
            })
            grouped['T'] = (pd.to_datetime(grouped['Year'].astype(str) + '-12-31') -
                            pd.Timestamp("2000-01-01")).dt.days

            grouped.to_csv(os.path.join(output_folder, file), index=False)


# ------------------------------ #
#        GHGRP PRODUCTION        #
# ------------------------------ #

def process_ghgrp_data():
    """
    Process GHGRP onshore production: filter rows, normalize well IDs to API14,
    join to production (API14, Year), and persist merged outputs.
    """
    ghgrp = pd.read_csv(ghgrp_path)
    ghgrp = ghgrp[ghgrp.industry_segment.notna()]
    ghgrp = ghgrp[ghgrp.industry_segment.str.contains('Onshore petroleum and natural gas production')]
    ghgrp = ghgrp[ghgrp.ch4_average_mole_fraction > 0]
    ghgrp = ghgrp[ghgrp.ch4_average_mole_fraction > ghgrp.co2_average_mole_fraction].reset_index(drop=True)

    # Extract county & state abbr from GHGRP sub_basin_county
    ghgrp['County']      = ghgrp['sub_basin_county'].str.extract(r'([A-Z\s]+),\s[A-Z]+\s\(\d+\)')
    ghgrp['State_abbr']  = ghgrp['sub_basin_county'].str.extract(r'([A-Z]+)\s\(\d+\)')

    # Map state abbreviations to full names
    state_abbreviations_to_names = {
        'AL':'Alabama','AK':'Alaska','AZ':'Arizona','AR':'Arkansas','CA':'California','CO':'Colorado','CT':'Connecticut',
        'DE':'Delaware','FL':'Florida','GA':'Georgia','HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa',
        'KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland','MA':'Massachusetts','MI':'Michigan',
        'MN':'Minnesota','MS':'Mississippi','MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire',
        'NJ':'New Jersey','NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma',
        'OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee',
        'TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'
    }
    ghgrp['State'] = ghgrp['State_abbr'].map(state_abbreviations_to_names).str.upper()

    # Load GHGRP well mapping (XLSB by year)
    wells_folder = os.path.join(root_path, 'ghgrp', 'EF_W_ONSHORE_WELLS')
    well_files = [os.path.join(wells_folder, f) for f in os.listdir(wells_folder) if f.endswith('.xlsb')]

    all_wells = []
    for file in tqdm(well_files, desc="GHGRP wells (xlsb)"):
        year = ''.join(filter(str.isdigit, os.path.basename(file)))
        df = pd.read_excel(file, usecols=['FACILITY_ID','WELL_ID_NUMBER','SUB_BASIN'], engine='pyxlsb')
        df['Year'] = int(year)
        all_wells.append(df)
    wells_df = pd.concat(all_wells, ignore_index=True)

    # Normalize WELL_ID_NUMBER to API14-like keys
    wells_df = wells_df.dropna(subset=['WELL_ID_NUMBER']).copy()
    wells_df['WELL_ID_NUMBER'] = wells_df['WELL_ID_NUMBER'].astype(str).str.replace('-', '', regex=False)
    wells_df['WELL_ID_NUMBER'] = wells_df['WELL_ID_NUMBER'].str.lstrip('0')

    def pad_api(api):
        """Pad to 13–14 chars based on original length (heuristics to align with API14)."""
        l = len(api)
        if   l == 12: return api.ljust(14, '0')
        elif l == 11: return api.ljust(13, '0')
        elif l == 10: return api.ljust(14, '0')
        elif l ==  9: return api.ljust(13, '0')
        else:         return api

    wells_df['API14'] = wells_df['WELL_ID_NUMBER'].apply(pad_api)

    # Load production (with Year, API14)
    prod_folder = os.path.join(root_path, 'wells_info_prod_per_basin')
    prod_files  = [os.path.join(prod_folder, f) for f in os.listdir(prod_folder)]
    all_prod = []
    for f in tqdm(prod_files, desc="Production (enriched)"):
        df = pd.read_csv(f)
        df = df[df['Year'] >= 2015]  # scope by year
        all_prod.append(df)
    prod_df = pd.concat(all_prod, ignore_index=True)
    prod_df['API14'] = prod_df['API14'].astype(str).str.lstrip('0')

    # Merge production with GHGRP wells by API14 + Year
    merged = pd.merge(prod_df, wells_df, on=['API14','Year'], how='inner')

    # Merge with GHGRP compositions/facility data
    ghgrp_merged = pd.merge(
        merged,
        ghgrp,
        left_on=['FACILITY_ID','Year','SUB_BASIN'],
        right_on=['facility_id','reporting_year','sub_basin_identifier'],
        how='inner'
    )

    # Clean columns & rename
    ghgrp_merged = ghgrp_merged.drop(columns=[
        'WELL_ID_NUMBER','facility_id','facility_name','basin_associated_with_facility','reporting_year',
        'sub_basin_county','sub_basin_formation_type','sub_basin_identifier','us_state','State_abbr','State'
    ], errors='ignore')

    ghgrp_merged = ghgrp_merged.rename(columns={
        'ch4_average_mole_fraction':'C1',
        'co2_average_mole_fraction':'CO2'
    })

    # Convert mole fractions to %
    ghgrp_merged['C1']  = ghgrp_merged['C1']  * 100.0
    ghgrp_merged['CO2'] = ghgrp_merged['CO2'] * 100.0
    ghgrp_merged = ghgrp_merged.drop_duplicates()

    out_csv = os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ghgrp_merged.to_csv(out_csv, index=False)
    print(f"Saved processed GHGRP production data: {out_csv}")


# ------------------------------ #
#       NEAREST NEIGHBORS        #
# ------------------------------ #

def nearest(row, other_gdf):
    """
    Find nearest geometry in other_gdf to the geometry of `row`.
    Returns (nearest_geometry, nearest_index). Emits a warning if geometry invalid.
    """
    if not row.geometry.is_valid:
        print(f"Invalid geometry: {explain_validity(row.geometry)}")
        return None, None
    try:
        nearest_geom = nearest_points(row.geometry, other_gdf.unary_union)[1]
        nearest_geom_idx = other_gdf.distance(nearest_geom).idxmin()
        return nearest_geom, nearest_geom_idx
    except Exception as e:
        print(f"Error finding nearest: {e}")
        return None, None


def get_nearest_distances(gdf1, gdf2):
    """
    For each geometry in gdf1, attach the nearest geometry in gdf2,
    plus the distance and attributes from that nearest gdf2 row.
    """
    if gdf1.crs != gdf2.crs:
        gdf1 = gdf1.to_crs(gdf2.crs)
    gdf2 = gdf2[gdf2.is_valid & ~gdf2.is_empty]

    nearest_geometries, nearest_indices, distances = [], [], []
    for _, row in gdf1.iterrows():
        geom, idx = nearest(row, gdf2)
        nearest_geometries.append(geom)
        nearest_indices.append(idx)
        distances.append(row.geometry.distance(geom) if geom is not None else None)

    gdf1['distance_to_nearest'] = distances
    gdf1['nearest_geom'] = nearest_geometries

    nearest_rows = gdf2.loc[nearest_indices].reset_index(drop=True)
    gdf1 = gdf1.reset_index(drop=True).join(nearest_rows, rsuffix='_gdf2')
    return gdf1


# ------------------------------ #
#  GHGRP PROCESSING FACILITIES   #
# ------------------------------ #

def process_ghgrp_processing_facilities(us_counties, basins):
    """
    Build processing-facility dataset with county & basin labels and
    annotate each capacity point with nearest facility and distance.
    """
    ghgrp = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_v2.csv'))
    ghgrp = ghgrp[ghgrp.industry_segment.notna()]
    ghgrp = ghgrp[ghgrp.industry_segment.str.contains('Onshore natural gas processing')]
    ghgrp = ghgrp[ghgrp.ch4_average_mole_fraction > 0].reset_index(drop=True)

    # Add lat/lon for facilities
    facilities_lat_lon = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_facilities_lat_lon.csv'))
    ghgrp = ghgrp.merge(facilities_lat_lon, on='facility_id')

    # Facility points → counties & basins
    geometry = [Point(lon, lat) for lat, lon in zip(ghgrp.latitude, ghgrp.longitude)]
    gdf = gpd.GeoDataFrame(ghgrp, geometry=geometry, crs=CRS_WGS84).to_crs(CRS_LOCAL)

    gdf = gpd.sjoin(gdf, us_counties.to_crs(CRS_LOCAL), predicate='within')
    gdf[['County', 'State']] = gdf['NAME'].str.extract(r'(.+)\s+\((.+)\)')
    gdf = gdf.drop(columns={'index_right'}, errors='ignore')
    gdf = gpd.sjoin(gdf, basins.to_crs(CRS_LOCAL), predicate='within')

    # Capacities layer (x=lon, y=lat columns as named in your CSV)
    capacities = pd.read_csv(os.path.join(root_path, 'ghgrp', 'processing_capacities.csv'),
                             usecols=['State','County','Cap_MMcfd','Plant_Flow','x','y'])
    geometry = [Point(lon, lat) for lat, lon in zip(capacities.y, capacities.x)]
    capacities_gdf = gpd.GeoDataFrame(capacities, geometry=geometry, crs='EPSG:3857').to_crs(CRS_LOCAL)

    # Attach nearest facility attributes to each capacity point
    enriched_gdf = get_nearest_distances(capacities_gdf, gdf)

    # Cleanup and persist
    enriched_gdf = enriched_gdf.drop(columns=['NAME','index_right','BASIN_CODE'], errors='ignore')
    enriched_gdf = enriched_gdf.rename(columns={'reporting_year':'Year', 'Monthly Gas':'Gas'})
    enriched_gdf = enriched_gdf.drop_duplicates().reset_index(drop=True)
    enriched_gdf['Count'] = 1

    out_csv = os.path.join(root_path, 'ghgrp_processing_new_weighted_variable_gdf.csv')
    enriched_gdf.to_csv(out_csv, index=False)
    print(f"Saved processed GHGRP processing plant data: {out_csv}")


# ------------------------------ #
#              RUN               #
# ------------------------------ #

if __name__ == "__main__":
    # 1) Shapefiles
    us_states, us_counties, basins = load_shapefiles()

    # 2) USGS
    usgs_gdf = clean_usgs_data(basins)

    # 3) Correlation plots (skips if Monthly Gas/Oil not present)
    correlation_analysis(usgs_gdf)

    # 4) Production time index
    add_T_to_production_files()

    # 5) GHGRP onshore production
    process_ghgrp_data()

    # 6) GHGRP processing facilities
    process_ghgrp_processing_facilities(us_counties, basins)

    print("All processing complete.")
