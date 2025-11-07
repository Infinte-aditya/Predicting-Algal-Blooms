import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load data from Step 01."""
    df = pd.read_csv('data/qgis_input/nal_sarovar_timeseries.csv', parse_dates=['date'])
    print(f"Loaded {len(df)} records from {df['date'].min().date()} to {df['date'].max().date()}")
    return df

def create_lag_features(df, columns, lags=[1, 2, 3, 4]):
    """Create lagged features (past values as predictors)."""
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

def create_rolling_features(df, columns, windows=[2, 4, 8]):
    """Create rolling statistics (moving averages, std)."""
    for col in columns:
        for window in windows:
            df[f'{col}_roll_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_roll_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
    return df

def create_temporal_features(df):
    """Extract time-based features."""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # Cyclical encoding (important for seasonality!)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    return df

def create_interaction_features(df):
    """Create interaction features between variables."""
    df['temp_ndvi_interaction'] = df['max_temp'] * df['ndvi']
    df['temp_rainfall_interaction'] = df['max_temp'] * df['rainfall_mm']
    df['ndvi_rainfall_interaction'] = df['ndvi'] * df['rainfall_mm']
    
    # Cumulative rainfall (previous 2 periods)
    df['cumulative_rain_2periods'] = df['rainfall_mm'].rolling(window=2, min_periods=1).sum()
    df['cumulative_rain_4periods'] = df['rainfall_mm'].rolling(window=4, min_periods=1).sum()
    
    return df

def create_rate_of_change(df, columns):
    """Calculate rate of change between periods."""
    for col in columns:
        df[f'{col}_change'] = df[col].diff()
        df[f'{col}_pct_change'] = df[col].pct_change().replace([np.inf, -np.inf], 0)
    return df

def create_target_variables(df):
    """Create multiple target variables for different prediction horizons."""
    # Next period bloom (16 days ahead)
    df['bloom_next_1'] = df['bloom'].shift(-1)
    df['bloom_next_2'] = df['bloom'].shift(-2)
    df['bloom_next_4'] = df['bloom'].shift(-4)  # ~2 months ahead
    
    # Risk score forecast
    df['risk_next_1'] = df['risk_score'].shift(-1)
    df['risk_next_2'] = df['risk_score'].shift(-2)
    
    return df

def remove_nan_rows(df):
    """Remove rows with NaN created by lag/rolling operations."""
    # Keep original date range for reference
    original_len = len(df)
    
    # Remove rows with NaN in feature columns (but keep last rows for prediction)
    feature_cols = [col for col in df.columns if col not in ['date', 'geometry', 'bloom_next_1', 'bloom_next_2', 'bloom_next_4', 'risk_next_1', 'risk_next_2']]
    
    # For training: remove rows where features OR targets are NaN
    df_train = df.dropna(subset=['bloom_next_1']).copy()
    
    # Fill remaining NaN in features with forward fill, then backward fill, then 0
    for col in feature_cols:
        if df_train[col].isna().any():
            df_train[col] = df_train[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Double check: replace any remaining NaN with 0
    df_train = df_train.fillna(0)
    
    # Replace infinite values with 0
    df_train = df_train.replace([np.inf, -np.inf], 0)
    
    print(f"Removed {original_len - len(df_train)} rows with NaN (from lag/rolling operations)")
    print(f"Training dataset: {len(df_train)} records")
    
    # Verify no NaN or inf remains
    nan_check = df_train.isna().sum().sum()
    inf_check = np.isinf(df_train.select_dtypes(include=[np.number]).values).sum()
    
    if nan_check > 0:
        print(f"⚠ WARNING: Still have {nan_check} NaN values")
    else:
        print(f"✓ No NaN values in final dataset")
    
    if inf_check > 0:
        print(f"⚠ WARNING: Still have {inf_check} infinite values")
    else:
        print(f"✓ No infinite values in final dataset")
    
    return df_train

def save_features(df, output_path='data/processed/ml_features.csv'):
    """Save feature-engineered dataset."""
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\n✅ Saved feature-engineered data to {output_path}")
    print(f"   Total features: {len(df.columns)}")
    print(f"   Total records: {len(df)}")
    
    # Save feature list
    feature_list = [col for col in df.columns if col not in ['date', 'geometry', 'bloom', 'bloom_next_1', 'bloom_next_2', 'bloom_next_4', 'risk_next_1', 'risk_next_2']]
    
    feature_info = pd.DataFrame({
        'feature': feature_list,
        'dtype': [df[col].dtype for col in feature_list]
    })
    feature_info.to_csv('data/processed/feature_list.csv', index=False)
    print(f"   Feature list saved to data/processed/feature_list.csv")

def main():
    print("="*60)
    print("FEATURE ENGINEERING FOR BLOOM PREDICTION")
    print("="*60)
    
    # Load data
    print("\n[1/8] Loading processed data...")
    df = load_processed_data()
    
    # Create features
    print("\n[2/8] Creating lag features (past values)...")
    df = create_lag_features(df, ['ndvi', 'max_temp', 'rainfall_mm'], lags=[1, 2, 3, 4])
    
    print("[3/8] Creating rolling window features...")
    df = create_rolling_features(df, ['ndvi', 'max_temp', 'rainfall_mm'], windows=[2, 4, 8])
    
    print("[4/8] Creating temporal features...")
    df = create_temporal_features(df)
    
    print("[5/8] Creating interaction features...")
    df = create_interaction_features(df)
    
    print("[6/8] Creating rate of change features...")
    df = create_rate_of_change(df, ['ndvi', 'max_temp', 'rainfall_mm'])
    
    print("[7/8] Creating target variables (future bloom labels)...")
    df = create_target_variables(df)
    
    print("[8/8] Cleaning and saving...")
    df_clean = remove_nan_rows(df)
    save_features(df_clean)
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE SUMMARY")
    print("="*60)
    
    feature_cols = [col for col in df_clean.columns if col not in ['date', 'geometry']]
    original_cols = ['ndvi', 'max_temp', 'rainfall_mm', 'bloom', 'risk_score', 'month']
    engineered_cols = [col for col in feature_cols if col not in original_cols]
    
    print(f"Original features: {len(original_cols)}")
    print(f"Engineered features: {len(engineered_cols)}")
    print(f"Total features: {len(feature_cols)}")
    print(f"\nTop engineered features:")
    for col in engineered_cols[:10]:
        print(f"  - {col}")
    
    # Show class balance
    bloom_counts = df_clean['bloom_next_1'].value_counts()
    print(f"\nTarget distribution (bloom_next_1):")
    print(f"  No bloom (0): {bloom_counts.get(0, 0)} ({bloom_counts.get(0, 0)/len(df_clean)*100:.1f}%)")
    print(f"  Bloom (1): {bloom_counts.get(1, 0)} ({bloom_counts.get(1, 0)/len(df_clean)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("✅ READY FOR MODEL TRAINING!")
    print("   Run: python src/03_train_models.py")
    print("="*60)

if __name__ == "__main__":
    main()