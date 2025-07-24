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



def load_shapefiles():
    us_states = gpd.read_file(os.path.join(shapefiles_path, 'cb_2018_us_state_20m', 'cb_2018_us_state_20m.shp'))
    us_counties = gpd.read_file(os.path.join(shapefiles_path, 'cb_2018_us_county_500k', 'cb_2018_us_county_500k.shp'))
    basins = gpd.read_file(os.path.join(shapefiles_path, 'basins_shapefiles'))

    index_to_deletes = []
    for i in range(len(us_counties)):
        county_name = us_counties['NAME'].iloc[i].upper()
        state_fp = us_counties['STATEFP'].iloc[i]
        if len(us_states[us_states.STATEFP == state_fp]) > 0:
            state_name = us_states[us_states.STATEFP == state_fp].reset_index(drop=True).NAME.iloc[0].upper()
            combined_name = f"{county_name} ({state_name})"
            us_counties.at[i, 'NAME'] = combined_name
            us_counties.at[i, 'COUNTY_NAME'] = county_name
            us_counties.at[i, 'STATE_NAME'] = state_name
        else:
            index_to_deletes.append(i)
    us_counties = us_counties.drop(index=index_to_deletes)
    return us_states, us_counties, basins


def clean_usgs_data(basins):
    usgs = pd.read_csv(usgs_path)
    usgs = usgs[usgs.LAT.notna() & usgs.LONG.notna() & usgs.FINAL_SAMPLING_DATE.notna()]
    usgs['Year'] = usgs['FINAL_SAMPLING_DATE'].str[-4:].astype(int)

    usgs_gdf = gpd.GeoDataFrame(usgs, geometry=gpd.points_from_xy(usgs.LONG, usgs.LAT), crs="EPSG:4326").to_crs(26914)
    usgs_gdf = usgs_gdf.rename(columns={'API': 'API14'})
    usgs_gdf['COUNTY'] = usgs_gdf['COUNTY'] + ' (' + usgs_gdf['STATE'] + ')'

    dates = []
    for raw_date in tqdm(usgs_gdf['FINAL_SAMPLING_DATE']):
        month, day, year = [int(x) for x in raw_date.split('/')]
        month, day = max(1, min(month, 12)), max(1, min(day, 28 if month == 2 else 30))
        dates.append(datetime(year, month, day))
    usgs_gdf['date'] = dates

    usgs_gdf = usgs_gdf.sort_values('date').reset_index(drop=True)
    usgs_gdf['epochs'] = (usgs_gdf['date'] - usgs_gdf['date'].iloc[0]).dt.days

    usgs_gdf = gpd.sjoin(usgs_gdf, basins.to_crs(26914), op='within')
    usgs_gdf['X'], usgs_gdf['Y'], usgs_gdf['T'] = usgs_gdf.geometry.x, usgs_gdf.geometry.y, usgs_gdf['epochs']

    drop_cols = ['ID', 'SOURCE','BLM ID','USGS ID','ALASKA_QUAD_NAMES','WELL NAME',
                 'PUBLICATION DATE','COUNTY','STATE','FINAL_DRILLING_DATE','FINAL_SAMPLING_DATE',
                 'SAMPLE DATE BEFORE COMPLETION', 'index_right', 'BASIN_CODE','DEPTH','AGE','FORMATION',
                 'geometry','FIELD','LAT','LONG']
    usgs_gdf = usgs_gdf.drop(columns=[c for c in drop_cols if c in usgs_gdf.columns])

    components = ['HE', 'CO2', 'H2', 'N2', 'H2S', 'AR', 'O2','C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
    usgs_gdf[components] = usgs_gdf[components].apply(pd.to_numeric, errors='coerce')

    usgs_gdf['geometry'] = gpd.points_from_xy(usgs_gdf['X'], usgs_gdf['Y'])
    usgs_gdf = gpd.GeoDataFrame(usgs_gdf, geometry='geometry', crs="EPSG:26914")

    key_columns = ['X', 'Y', 'Year', 'date', 'T', 'BASIN_NAME']
    agg_df = usgs_gdf.groupby(key_columns).agg({col: np.nanmean for col in components}).reset_index()
    agg_df.to_csv(os.path.join(root_path, 'usgs', 'usgs_processed_with_nanapis.csv'), index=False)
    return usgs_gdf


def correlation_analysis(usgs_gdf):
    df = usgs_gdf.copy()
    df["GOR"] = df["Monthly Gas"] / df["Monthly Oil"]
    df["Log GOR"] = np.log(df["GOR"])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["Log GOR"], inplace=True)

    components = ['C1', 'C2', 'C3', 'N-C4', 'I-C4', 'N-C5', 'I-C5', 'C6+']
    palette = sns.color_palette("crest", n_colors=len(components))[::-1]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i, (comp, color) in enumerate(zip(components, palette)):
        ax = axes[i]
        mask = df["Log GOR"].notna() & df[comp].notna()
        if mask.sum() > 1:
            ax.scatter(df.loc[mask, "Log GOR"], df.loc[mask, comp], alpha=0.3, color=color, edgecolors="none")
            slope, intercept, r_value, _, _ = linregress(df.loc[mask, "Log GOR"], df.loc[mask, comp])
            x_vals = np.linspace(df["Log GOR"].min(), df["Log GOR"].max(), 100)
            ax.plot(x_vals, slope * x_vals + intercept, color="black")
            ax.text(0.05, 0.9, f"r={r_value:.3f}", transform=ax.transAxes, bbox=dict(facecolor="white", alpha=0.5))
        ax.set_title(comp, fontsize=10, fontweight="bold")
        ax.set_xlabel("Log GOR")
        ax.set_ylabel(f"% {comp}")

    plt.tight_layout()
    plt.savefig('figures_out/correlation_plots.png', dpi=300)
    plt.show()


def add_T_to_production_files():
    input_folder = os.path.join(root_path, 'wells_prod_jan_26')
    output_folder = os.path.join(root_path, 'wells_info_prod_per_basin')
    os.makedirs(output_folder, exist_ok=True)

    for file in tqdm(os.listdir(input_folder)):
        if file.endswith('.csv') and 'Anadarko' in file:
            df = pd.read_csv(os.path.join(input_folder, file))
            df['geometry'] = df.apply(lambda row: Point(row['Surface Hole Longitude (WGS84)'], row['Surface Hole Latitude (WGS84)']), axis=1)
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326").to_crs(26914)
            gdf['X'], gdf['Y'] = gdf.geometry.x, gdf.geometry.y
            gdf['Monthly Production Date'] = pd.to_datetime(gdf['Monthly Production Date'])
            gdf['Year'] = gdf['Monthly Production Date'].dt.year
            grouped = gdf.groupby(['API14', 'X', 'Y', 'Year'], as_index=False).agg({
                'Monthly Gas': 'sum',
                'Monthly Oil': 'sum',
                'BASIN_NAME': 'first'
            })
            grouped['T'] = (pd.to_datetime(grouped['Year'].astype(str) + '-12-31') - pd.Timestamp("2000-01-01")).dt.days
            grouped.to_csv(os.path.join(output_folder, file), index=False)


def process_ghgrp_data():
    ghgrp = pd.read_csv(ghgrp_path)
    ghgrp = ghgrp[ghgrp.industry_segment.notna()]
    ghgrp = ghgrp[ghgrp.industry_segment.str.contains('Onshore petroleum and natural gas production')]
    ghgrp = ghgrp[ghgrp.ch4_average_mole_fraction > 0]
    ghgrp = ghgrp[ghgrp.ch4_average_mole_fraction > ghgrp.co2_average_mole_fraction].reset_index(drop=True)

    ghgrp['County'] = ghgrp['sub_basin_county'].str.extract(r'([A-Z\s]+),\s[A-Z]+\s\(\d+\)')
    ghgrp['State_abbr'] = ghgrp['sub_basin_county'].str.extract(r'([A-Z]+)\s\(\d+\)')
    state_abbreviations_to_names = {
        'AL': 'Alabama',
        'AK': 'Alaska',
        'AZ': 'Arizona',
        'AR': 'Arkansas',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'HI': 'Hawaii',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'IA': 'Iowa',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'ME': 'Maine',
        'MD': 'Maryland',
        'MA': 'Massachusetts',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MS': 'Mississippi',
        'MO': 'Missouri',
        'MT': 'Montana',
        'NE': 'Nebraska',
        'NV': 'Nevada',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NY': 'New York',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VT': 'Vermont',
        'VA': 'Virginia',
        'WA': 'Washington',
        'WV': 'West Virginia',
        'WI': 'Wisconsin',
        'WY': 'Wyoming'
    }
    ghgrp['State'] = ghgrp['State_abbr'].map(state_abbrev_to_name).str.upper()

    wells_folder = os.path.join(root_path, 'ghgrp', 'EF_W_ONSHORE_WELLS')
    well_files = [os.path.join(wells_folder, f) for f in os.listdir(wells_folder) if f.endswith('.xlsb')]

    all_wells = []
    for file in tqdm(well_files):
        year = ''.join(filter(str.isdigit, os.path.basename(file)))
        df = pd.read_excel(file, usecols=['FACILITY_ID', 'WELL_ID_NUMBER', 'SUB_BASIN'], engine='pyxlsb')
        df['Year'] = int(year)
        all_wells.append(df)
    wells_df = pd.concat(all_wells, ignore_index=True)

    wells_df = wells_df.dropna(subset=['WELL_ID_NUMBER'])
    wells_df['WELL_ID_NUMBER'] = wells_df['WELL_ID_NUMBER'].astype(str).str.replace('-', '')
    wells_df['WELL_ID_NUMBER'] = wells_df['WELL_ID_NUMBER'].str.lstrip('0')

    def pad_api(api):
        l = len(api)
        return api.ljust(14, '0') if l == 12 else api.ljust(13, '0') if l == 11 else api.ljust(14, '0') if l == 10 else api.ljust(13, '0') if l == 9 else api

    wells_df['API14'] = wells_df['WELL_ID_NUMBER'].apply(pad_api)

    prod_folder = os.path.join(root_path, 'wells_info_prod_per_basin')
    prod_files = [os.path.join(prod_folder, f) for f in os.listdir(prod_folder)]

    all_prod = []
    for f in tqdm(prod_files):
        df = pd.read_csv(f)
        df = df[df['Year'] >= 2015]
        all_prod.append(df)
    prod_df = pd.concat(all_prod, ignore_index=True)
    prod_df['API14'] = prod_df['API14'].astype(str).str.lstrip('0')

    merged = pd.merge(prod_df, wells_df, left_on=['API14', 'Year'], right_on=['API14', 'Year'], how='inner')
    ghgrp_merged = pd.merge(merged, ghgrp, left_on=['FACILITY_ID', 'Year', 'SUB_BASIN'], right_on=['facility_id', 'reporting_year', 'sub_basin_identifier'])

    ghgrp_merged = ghgrp_merged.drop(columns=[
        'WELL_ID_NUMBER', 'facility_id', 'facility_name', 'basin_associated_with_facility', 'reporting_year',
        'sub_basin_county', 'sub_basin_formation_type', 'sub_basin_identifier', 'us_state', 'State_abbr', 'State'
    ], errors='ignore')

    ghgrp_merged = ghgrp_merged.rename(columns={
        'ch4_average_mole_fraction': 'C1',
        'co2_average_mole_fraction': 'CO2'
    })

    ghgrp_merged['C1'] *= 100
    ghgrp_merged['CO2'] *= 100
    ghgrp_merged = ghgrp_merged.drop_duplicates()

    ghgrp_merged.to_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_production_processed.csv'), index=False)
    print("Saved processed GHGRP production data.")


def nearest(row, other_gdf):
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
    if gdf1.crs != gdf2.crs:
        gdf1 = gdf1.to_crs(gdf2.crs)
    gdf2 = gdf2[gdf2.is_valid & ~gdf2.is_empty]
    nearest_geometries, nearest_indices, distances = [], [], []
    for _, row in gdf1.iterrows():
        geom, idx = nearest(row, gdf2)
        nearest_geometries.append(geom)
        nearest_indices.append(idx)
        distances.append(row.geometry.distance(geom) if geom else None)
    gdf1['distance_to_nearest'] = distances
    gdf1['nearest_geom'] = nearest_geometries
    nearest_rows = gdf2.loc[nearest_indices].reset_index(drop=True)
    gdf1 = gdf1.reset_index(drop=True).join(nearest_rows, rsuffix='_gdf2')
    return gdf1

def process_ghgrp_processing_facilities(us_counties, basins):
    ghgrp = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_v2.csv'))
    ghgrp = ghgrp[ghgrp.industry_segment.notna()]
    ghgrp = ghgrp[ghgrp.industry_segment.str.contains('Onshore natural gas processing')]
    ghgrp = ghgrp[ghgrp.ch4_average_mole_fraction > 0].reset_index(drop=True)

    facilities_lat_lon = pd.read_csv(os.path.join(root_path, 'ghgrp', 'ghgrp_facilities_lat_lon.csv'))
    ghgrp = ghgrp.merge(facilities_lat_lon, on='facility_id')

    geometry = [Point(lon, lat) for lat, lon in zip(ghgrp.latitude, ghgrp.longitude)]
    gdf = gpd.GeoDataFrame(ghgrp, geometry=geometry, crs='EPSG:4326').to_crs(26914)
    gdf = gpd.sjoin(gdf, us_counties.to_crs(26914), op='within')
    gdf[['County', 'State']] = gdf['NAME'].str.extract(r'(.+)\s+\((.+)\)')
    gdf = gdf.drop(columns={'index_right'})
    gdf = gpd.sjoin(gdf, basins.to_crs(26914), op='within')

    capacities = pd.read_csv(os.path.join(root_path, 'ghgrp', 'processing_capacities.csv'),
                             usecols=['State', 'County', 'Cap_MMcfd', 'Plant_Flow', 'x', 'y'])
    geometry = [Point(lon, lat) for lat, lon in zip(capacities.y, capacities.x)]
    capacities_gdf = gpd.GeoDataFrame(capacities, geometry=geometry, crs='EPSG:3857').to_crs(26914)

    enriched_gdf = get_nearest_distances(capacities_gdf, gdf)

    enriched_gdf = enriched_gdf.drop(columns=['NAME', 'index_right', 'BASIN_CODE'], errors='ignore')
    enriched_gdf = enriched_gdf.rename(columns={'reporting_year': 'Year', 'Monthly Gas': 'Gas'})
    enriched_gdf = enriched_gdf.drop_duplicates().reset_index(drop=True)
    enriched_gdf['Count'] = 1

    enriched_gdf.to_csv(os.path.join(root_path, 'ghgrp_processing_new_weighted_variable_gdf.csv'), index=False)
    print("Saved processed GHGRP processing plant data.")



us_states, us_counties, basins = load_shapefiles()
usgs_gdf = clean_usgs_data(basins)
correlation_analysis(usgs_gdf)
add_T_to_production_files()
process_ghgrp_data()
process_ghgrp_processing_facilities(us_counties, basins)
print("All processing complete.")

