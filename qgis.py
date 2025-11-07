# 01_prepare_for_qgis.py (OPTIMIZED)
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

# Configuration
NAL_SAROVAR_LAT = 22.77
NAL_SAROVAR_LON = 72.05
RESAMPLE_PERIOD = '16D'
BLOOM_THRESHOLDS = {
    'ndvi': 0.40,
    'temp': 25,
    'rainfall': 50,
    'months': [10, 11, 12]
}

def load_and_prepare_data():
    """Load CSVs with optimized settings."""
    # Load NDVI data
    ndvi = pd.read_csv('data/ndvi_nal_sarovar.csv')
    ndvi['date'] = pd.to_datetime(ndvi['date'])
    ndvi = ndvi[['date', 'ndvi']]
    
    # Handle duplicate headers in temp file
    temp = pd.read_csv('data/temp_ahmedabad.csv')
    # Get the first 'Date' and first 'Temp Max' column
    date_col = [col for col in temp.columns if 'Date' in col][0]
    temp_col = [col for col in temp.columns if 'Temp Max' in col][0]
    
    temp = temp[[date_col, temp_col]].copy()
    temp.columns = ['date', 'max_temp']
    temp['date'] = pd.to_datetime(temp['date'], errors='coerce')
    temp = temp.dropna(subset=['date'])
    temp['max_temp'] = pd.to_numeric(temp['max_temp'], errors='coerce')
    
    # Handle rainfall data with proper column names
    rain = pd.read_csv('data/rainfall_surendranagar.csv')
    rain['date'] = pd.to_datetime(rain['Date'], errors='coerce')
    rain = rain.dropna(subset=['date'])
    rain['rainfall_mm'] = pd.to_numeric(rain['Avg_rainfall'], errors='coerce')
    rain = rain[['date', 'rainfall_mm']]
    
    return ndvi, temp, rain

def resample_timeseries(temp, rain, period=RESAMPLE_PERIOD):
    """Resample temperature and rainfall to specified period."""
    # Ensure datetime index and remove any remaining NaT
    temp = temp.dropna(subset=['date']).copy()
    rain = rain.dropna(subset=['date']).copy()
    
    # Convert numeric columns
    temp['max_temp'] = pd.to_numeric(temp['max_temp'], errors='coerce')
    rain['rainfall_mm'] = pd.to_numeric(rain['rainfall_mm'], errors='coerce')
    
    # Resample with proper datetime index
    temp_resampled = (temp.set_index('date')
                      .sort_index()
                      .resample(period)
                      .mean()
                      .reset_index())
    
    rain_resampled = (rain.set_index('date')
                      .sort_index()
                      .resample(period)
                      .sum()
                      .reset_index())
    
    return temp_resampled, rain_resampled

def merge_datasets(ndvi, temp, rain):
    """Efficiently merge all datasets."""
    return (ndvi.merge(temp, on='date', how='left')
            .merge(rain, on='date', how='left'))

def clean_data(df):
    """Clean and interpolate missing values."""
    # Interpolate temperature (linear interpolation)
    df['max_temp'] = df['max_temp'].interpolate(method='linear', limit_direction='both')
    
    # Fill rainfall NaN with 0 (no rain)
    df['rainfall_mm'] = df['rainfall_mm'].fillna(0)
    
    # Extract month for bloom detection
    df['month'] = df['date'].dt.month
    
    return df

def calculate_bloom_risk(df, thresholds=BLOOM_THRESHOLDS):
    """Calculate bloom events and risk scores using vectorized operations."""
    # Bloom detection (vectorized)
    bloom_conditions = (
        (df['ndvi'] > thresholds['ndvi']) &
        (df['max_temp'] > thresholds['temp']) &
        (df['month'].isin(thresholds['months']))
    )
    df['bloom'] = bloom_conditions.astype(np.int8)
    
    # Risk score calculation (vectorized)
    ndvi_risk = (df['ndvi'] > thresholds['ndvi']).astype(float) * 0.4
    temp_risk = (df['max_temp'] > thresholds['temp']).astype(float) * 0.4
    rain_risk = (df['rainfall_mm'] > thresholds['rainfall']).astype(float) * 0.2
    
    df['risk_score'] = ndvi_risk + temp_risk + rain_risk
    
    return df

def create_geodataframe(df, lat=NAL_SAROVAR_LAT, lon=NAL_SAROVAR_LON):
    """Create GeoDataFrame with point geometry.
    
    Options:
    1. Single point (stacked): All timestamps at same location
    2. Spatial grid: Spread points in a grid pattern for visualization
    3. Time-based offset: Slight offset based on time for distinction
    """
    
    # OPTION 1: Keep as single point (original - will stack in QGIS)
    geometry = gpd.points_from_xy([lon] * len(df), [lat] * len(df))
    
    # # OPTION 2: Create a small spatial grid for better visualization
    # # Spread points in a 0.1 degree grid around the center
    # n = len(df)
    # grid_size = int(np.ceil(np.sqrt(n)))
    # offset = 0.01  # 0.01 degrees ≈ 1.1 km
    
    # lons = []
    # lats = []
    # for i in range(n):
    #     row = i // grid_size
    #     col = i % grid_size
    #     lons.append(lon + (col - grid_size/2) * offset)
    #     lats.append(lat + (row - grid_size/2) * offset)
    
    # geometry = gpd.points_from_xy(lons, lats)
    
    
    # OPTION 3: Alternative - Circular pattern (uncomment to use)
    # angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    # radius = 0.05  # 0.05 degrees ≈ 5.5 km
    # lons = lon + radius * np.cos(angles)
    # lats = lat + radius * np.sin(angles)
    # geometry = gpd.points_from_xy(lons, lats)
    
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf

def export_data(gdf, output_dir='data/qgis_input'):
    """Export to shapefile and CSV with proper time-series format."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    shp_path = f"{output_dir}/nal_sarovar_timeseries.shp"
    csv_path = f"{output_dir}/nal_sarovar_timeseries.csv"
    gpkg_path = f"{output_dir}/nal_sarovar_timeseries.gpkg"
    
    # Prepare data with DATETIME fields (not strings!)
    gdf_export = gdf.copy()
    
    # Keep as datetime objects for proper temporal support
    gdf_export['date_start'] = gdf_export['date']
    gdf_export['date_end'] = gdf_export['date'] + pd.Timedelta(days=16)
    
    # For shapefile: Use SHORT field names (10 char limit!)
    gdf_export['dt_start'] = gdf_export['date']
    gdf_export['dt_end'] = gdf_export['date_end']
    gdf_export['temp'] = gdf_export['max_temp']
    gdf_export['rain'] = gdf_export['rainfall_mm']
    
    # Select columns for shapefile (with short names)
    shp_cols = ['dt_start', 'dt_end', 'ndvi', 'temp', 'rain', 
                'bloom', 'risk_score', 'month', 'geometry']
    gdf_shp = gdf_export[shp_cols].copy()
    
    # Export shapefile
    gdf_shp.to_file(shp_path)
    
    # Export GeoPackage with full datetime support
    gpkg_cols = ['date', 'date_start', 'date_end', 'ndvi', 'max_temp', 'rainfall_mm',
                 'bloom', 'risk_score', 'month', 'geometry']
    gdf_gpkg = gdf_export[gpkg_cols].copy()
    gdf_gpkg.to_file(gpkg_path, driver='GPKG')
    
    # CSV export
    gdf.to_csv(csv_path, index=False)
    
    # Summary CSV
    summary_path = f"{output_dir}/bloom_summary.csv"
    summary = gdf[['date', 'ndvi', 'max_temp', 'rainfall_mm', 'bloom', 'risk_score']].copy()
    summary.to_csv(summary_path, index=False)
    
    return shp_path, gpkg_path, csv_path, summary_path

def main():
    """Main execution pipeline."""
    print("Loading data...")
    ndvi, temp, rain = load_and_prepare_data()
    
    print("Resampling timeseries...")
    temp16, rain16 = resample_timeseries(temp, rain)
    
    print("Merging datasets...")
    df = merge_datasets(ndvi, temp16, rain16)
    
    print("Cleaning data...")
    df = clean_data(df)
    
    print("Calculating bloom risk...")
    df = calculate_bloom_risk(df)
    
    print("Creating GeoDataFrame...")
    gdf = create_geodataframe(df)
    
    print("Exporting data...")
    shp_path, gpkg_path, csv_path, summary_path = export_data(gdf)
    
    # Summary statistics
    bloom_count = df['bloom'].sum()
    avg_risk = df['risk_score'].mean()
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    
    print("\n" + "="*50)
    print("PROCESSING COMPLETE!")
    print("="*50)
    print(f"Total records: {len(df)}")
    print(f"Date range: {date_range}")
    print(f"Bloom events detected: {bloom_count}")
    print(f"Average risk score: {avg_risk:.3f}")
    print(f"\nOutputs:")
    print(f"  - Shapefile: {shp_path}")
    print(f"  - GeoPackage: {gpkg_path} (RECOMMENDED - no field limits)")
    print(f"  - CSV: {csv_path}")
    print(f"  - Summary: {summary_path}")
    print("\n" + "="*50)
    print("QGIS TEMPORAL CONTROLLER SETUP:")
    print("="*50)
    print("*** USE GEOPACKAGE (.gpkg) - Better than shapefile! ***")
    print("\n1. Load nal_sarovar_timeseries.gpkg in QGIS")
    print("2. Right-click layer → Properties → Temporal")
    print("3. Enable 'Dynamic Temporal Control'")
    print("4. Configuration: 'Single Field with Date/Time'")
    print("5. Field: 'date_start' (GeoPackage)")
    print("   OR Field: 'dt_start' (if using Shapefile)")
    print("6. Click OK")
    print("\n7. Open Temporal Controller panel:")
    print("   View → Panels → Temporal Controller")
    print("8. Set range to match your data dates")
    print("9. Step: 16 days")
    print("10. Click Play ▶️")
    print("\n" + "="*50)
    print("STYLING TIPS:")
    print("="*50)
    print("Layer Properties → Symbology → Categorized")
    print("  Column: 'bloom'")
    print("  0 = Green circle (No bloom)")
    print("  1 = Red circle (Bloom detected)")
    print("  Size: 8-12 pt for visibility")
    print("\nAlternative: Use 'Graduated' with 'risk_score'")
    print("  Color ramp: Yellow → Orange → Red")
    print("="*50)

if __name__ == "__main__":
    main()