import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

base_folder = '/content/drive/Shared drives/Enverus US'
output_folder = os.path.join(base_folder, 'Processed Data')
raw_wells_production_folder = os.path.join(base_folder, 'Well Monthly Production - Anadarko')
raw_wells_headers_folder = os.path.join(base_folder, 'Well Headers - Anadarko')
wells_production_folder = os.path.join(output_folder, 'Well Monthly Production with useful columns - Anadarko')
wells_headers_folder = os.path.join(output_folder, 'Well Headers with useful columns - Anadarko')

os.makedirs(output_folder, exist_ok=True)
os.makedirs(wells_production_folder, exist_ok=True)
os.makedirs(wells_headers_folder, exist_ok=True)

production_csv_files = [f for f in os.listdir(raw_wells_production_folder)]
for file in tqdm(production_csv_files, desc="Processing CSV files"):
    file_path = os.path.join(raw_wells_production_folder, file)
    processed_df = pd.read_csv(
        file_path,
        usecols=['API/UWI', 'Monthly Oil', 'Monthly Gas', 'Monthly Production Date']
    )
    processed_df = processed_df[~pd.isna(processed_df['API/UWI'])]
    processed_df['API/UWI'] = processed_df['API/UWI'].astype(str)
    processed_df = processed_df[processed_df['API/UWI'] != '0']
    output_path = os.path.join(wells_production_folder, file)
    processed_df.to_csv(output_path, index=False)

headers_csv_files = [f for f in os.listdir(raw_wells_headers_folder) if f.lower().endswith('.csv')]
for file in tqdm(headers_csv_files, desc="Processing CSV files"):
    file_path = os.path.join(raw_wells_headers_folder, file)
    processed_df = pd.read_csv(
        file_path,
        usecols=['API14', 'Surface Hole Latitude (WGS84)', 'Surface Hole Longitude (WGS84)']
    )
    processed_df = processed_df[~pd.isna(processed_df['API14'])]
    processed_df['API14'] = processed_df['API14'].astype(str)
    processed_df = processed_df[processed_df['API14'] != '0']
    output_path = os.path.join(wells_headers_folder, file)
    processed_df.to_csv(output_path, index=False)

def process_well_headers(headers_folder, basin_gdf, output_folder):
    output_file = os.path.join(output_folder, 'wellheaderswithbasins.csv')
    headers_files = [f for f in os.listdir(headers_folder) if f.lower().endswith('.csv')]
    headers_df = pd.concat([
        pd.read_csv(os.path.join(headers_folder, file)) for file in tqdm(headers_files, desc="Reading headers")
    ], ignore_index=True)
    headers_gdf = gpd.GeoDataFrame(
        headers_df,
        geometry=[Point(xy) for xy in zip(headers_df['Surface Hole Longitude (WGS84)'], headers_df['Surface Hole Latitude (WGS84)'])],
        crs='EPSG:4326'
    )
    headers_gdf = gpd.sjoin(headers_gdf.to_crs(26914), basin_gdf.to_crs(26914), how='inner', predicate='within')
    headers_gdf[['API14', 'Surface Hole Latitude (WGS84)', 'Surface Hole Longitude (WGS84)', 'BASIN_NAME']].to_csv(output_file, index=False)
    return output_file

def process_production_files(production_folder, headers_file, output_folder):
    headers_df = pd.read_csv(headers_file)
    production_files = [f for f in os.listdir(production_folder) if f.lower().endswith('.csv')]
    for file in tqdm(production_files, desc="Processing production files"):
        production_df = pd.read_csv(os.path.join(production_folder, file))
        production_df['API/UWI'] = production_df['API/UWI'].astype(str)
        headers_df['API14'] = headers_df['API14'].astype(str)
        merged_df = pd.merge(
            production_df,
            headers_df,
            left_on='API/UWI',
            right_on='API14',
            how='left'
        )
        output_file = os.path.join(output_folder, f"processed_{file}")
        merged_df.to_csv(output_file, index=False)

def split_by_basin(input_folder, output_folder):
    input_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.csv')]
    for file in tqdm(input_files, desc="Splitting by BASIN_NAME"):
        df = pd.read_csv(os.path.join(input_folder, file))
        for basin_name, group in df.groupby('BASIN_NAME'):
            basin_filename = f"{basin_name.replace(' ', '_')}_{file}"
            basin_output_path = os.path.join(output_folder, basin_filename)
            group.to_csv(basin_output_path, index=False)

def concat_by_basin(input_folder, final_output_folder, basins_gdf, processed_files=None):
    if processed_files is None:
        processed_files = []
    for basin_name in tqdm(basins_gdf.BASIN_NAME.unique(), desc="Processing basins"):
        formatted_name = basin_name.replace(' ', '_')
        matching_files = [
            f for f in os.listdir(input_folder)
            if f.lower().endswith('.csv') and f.split('_processed')[0] == formatted_name
        ]
        matching_files = [f for f in matching_files if f not in processed_files]
        if not matching_files:
            continue
        dfs = [pd.read_csv(os.path.join(input_folder, f)) for f in matching_files]
        processed_files.extend(matching_files)
        final_df = pd.concat(dfs, ignore_index=True)
        final_df['BASIN_NAME'] = basin_name
        final_file = os.path.join(final_output_folder, f"{basin_name}_final.csv")
        final_df.to_csv(final_file, index=False)
    return processed_files

headers_output_folder = os.path.join(output_folder, 'Well Headers with Basins')
merged_output_folder = os.path.join(output_folder, 'Well Monthly Production with Headers and Basins')
split_output_folder = os.path.join(output_folder, 'Well Monthly Production with Headers and Basins split by Basin')
final_output_folder = os.path.join(output_folder, 'Well Monthly Production with Headers and Basins concatenated by Basin')
os.makedirs(headers_output_folder, exist_ok=True)
os.makedirs(merged_output_folder, exist_ok=True)
os.makedirs(split_output_folder, exist_ok=True)
os.makedirs(final_output_folder, exist_ok=True)
basin_gdf_path = os.path.join(base_folder, 'Basins_Shapefile')
basin_gdf = gpd.read_file(basin_gdf_path)
headers_file = process_well_headers(wells_headers_folder, basin_gdf, headers_output_folder)
process_production_files(wells_production_folder, headers_file, merged_output_folder)
split_by_basin(merged_output_folder, split_output_folder)
concat_by_basin(split_output_folder, final_output_folder, basin_gdf)