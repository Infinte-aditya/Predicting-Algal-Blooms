import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_best_model(model_name='xgboost'):
    """Load the best trained model."""
    model = joblib.load(f'data/models/{model_name}.joblib')
    feature_cols = joblib.load('data/models/feature_columns.joblib')
    print(f"Loaded {model_name} model with {len(feature_cols)} features")
    return model, feature_cols

def load_historical_data():
    """Load historical data with features."""
    df = pd.read_csv('data/processed/ml_features.csv', parse_dates=['date'])
    return df

def get_seasonal_baseline(df_historical, month):
    """Get historical average values for a given month."""
    month_data = df_historical[df_historical['month'] == month]
    
    if len(month_data) > 0:
        return {
            'ndvi': month_data['ndvi'].mean(),
            'max_temp': month_data['max_temp'].mean(),
            'rainfall_mm': month_data['rainfall_mm'].mean(),
            'ndvi_std': month_data['ndvi'].std(),
            'temp_std': month_data['max_temp'].std(),
            'rain_std': month_data['rainfall_mm'].std()
        }
    else:
        # Fallback to overall average
        return {
            'ndvi': df_historical['ndvi'].mean(),
            'max_temp': df_historical['max_temp'].mean(),
            'rainfall_mm': df_historical['rainfall_mm'].mean(),
            'ndvi_std': df_historical['ndvi'].std(),
            'temp_std': df_historical['max_temp'].std(),
            'rain_std': df_historical['rainfall_mm'].std()
        }

def create_future_dates(last_date, periods=24):  # Increased to 24 (1 year)
    """Generate future dates (16-day intervals)."""
    future_dates = []
    current_date = last_date + timedelta(days=16)
    
    for i in range(periods):
        future_dates.append(current_date)
        current_date += timedelta(days=16)
    
    return pd.DataFrame({'date': future_dates})

def generate_realistic_forecast(df_historical, future_dates, model, feature_cols, scenario='normal'):
    """Generate realistic forecasts using historical seasonal patterns."""
    
    print(f"\n   Generating {scenario.upper()} scenario (seasonal-aware)...")
    
    df_future = future_dates.copy()
    df_future['month'] = df_future['date'].dt.month
    df_future['year'] = df_future['date'].dt.year
    
    # Scenario multipliers
    if scenario == 'wet':
        rain_mult = 1.5
        temp_adj = -2
        ndvi_mult = 1.15
    elif scenario == 'dry':
        rain_mult = 0.5
        temp_adj = 3
        ndvi_mult = 0.85
    else:  # normal
        rain_mult = 1.0
        temp_adj = 0
        ndvi_mult = 1.0
    
    # Generate realistic values based on historical monthly averages
    for idx, row in df_future.iterrows():
        month = row['month']
        baseline = get_seasonal_baseline(df_historical, month)
        
        # Add realistic variation using historical std
        df_future.at[idx, 'ndvi'] = np.clip(
            baseline['ndvi'] * ndvi_mult + np.random.normal(0, baseline['ndvi_std'] * 0.3),
            0, 1
        )
        
        df_future.at[idx, 'max_temp'] = np.clip(
            baseline['max_temp'] + temp_adj + np.random.normal(0, baseline['temp_std'] * 0.3),
            10, 50
        )
        
        df_future.at[idx, 'rainfall_mm'] = np.clip(
            baseline['rainfall_mm'] * rain_mult + np.random.normal(0, baseline['rain_std'] * 0.3),
            0, 500
        )
    
    # Add temporal features
    df_future['quarter'] = df_future['date'].dt.quarter
    df_future['day_of_year'] = df_future['date'].dt.dayofyear
    
    # Cyclical encoding
    df_future['month_sin'] = np.sin(2 * np.pi * df_future['month'] / 12)
    df_future['month_cos'] = np.cos(2 * np.pi * df_future['month'] / 12)
    df_future['day_sin'] = np.sin(2 * np.pi * df_future['day_of_year'] / 365)
    df_future['day_cos'] = np.cos(2 * np.pi * df_future['day_of_year'] / 365)
    
    # Create lag features iteratively
    # For first period, use last historical values
    last_row = df_historical.iloc[-1]
    
    for idx in range(len(df_future)):
        row = df_future.iloc[idx]
        
        # For lag1, use previous period (either historical or just-predicted)
        if idx == 0:
            for col in ['ndvi', 'max_temp', 'rainfall_mm']:
                df_future.at[idx, f'{col}_lag1'] = last_row[col]
                for lag in [2, 3, 4]:
                    df_future.at[idx, f'{col}_lag{lag}'] = last_row.get(f'{col}_lag{lag-1}', last_row[col])
        else:
            for col in ['ndvi', 'max_temp', 'rainfall_mm']:
                df_future.at[idx, f'{col}_lag1'] = df_future.at[idx-1, col]
                df_future.at[idx, f'{col}_lag2'] = df_future.at[idx-1, f'{col}_lag1']
                df_future.at[idx, f'{col}_lag3'] = df_future.at[idx-1, f'{col}_lag2']
                df_future.at[idx, f'{col}_lag4'] = df_future.at[idx-1, f'{col}_lag3']
    
    # Rolling features - use expanding window from historical
    for col in ['ndvi', 'max_temp', 'rainfall_mm']:
        for window in [2, 4, 8]:
            # Get historical values for rolling calc
            hist_values = df_historical[col].tail(window).values
            
            for idx in range(len(df_future)):
                # Combine historical + predicted values
                if idx < window:
                    combined = np.concatenate([hist_values[-(window-idx):], df_future[col].iloc[:idx+1].values])
                else:
                    combined = df_future[col].iloc[idx-window+1:idx+1].values
                
                df_future.at[idx, f'{col}_roll_mean_{window}'] = np.mean(combined)
                df_future.at[idx, f'{col}_roll_std_{window}'] = np.std(combined)
    
    # Interaction features
    df_future['temp_ndvi_interaction'] = df_future['max_temp'] * df_future['ndvi']
    df_future['temp_rainfall_interaction'] = df_future['max_temp'] * df_future['rainfall_mm']
    df_future['ndvi_rainfall_interaction'] = df_future['ndvi'] * df_future['rainfall_mm']
    
    # Cumulative rainfall
    for idx in range(len(df_future)):
        if idx < 2:
            hist_rain = df_historical['rainfall_mm'].tail(2).sum()
            future_rain = df_future['rainfall_mm'].iloc[:idx+1].sum()
            df_future.at[idx, 'cumulative_rain_2periods'] = hist_rain + future_rain
        else:
            df_future.at[idx, 'cumulative_rain_2periods'] = df_future['rainfall_mm'].iloc[idx-1:idx+1].sum()
        
        if idx < 4:
            hist_rain = df_historical['rainfall_mm'].tail(4).sum()
            future_rain = df_future['rainfall_mm'].iloc[:idx+1].sum()
            df_future.at[idx, 'cumulative_rain_4periods'] = hist_rain + future_rain
        else:
            df_future.at[idx, 'cumulative_rain_4periods'] = df_future['rainfall_mm'].iloc[idx-3:idx+1].sum()
    
    # Rate of change
    for col in ['ndvi', 'max_temp', 'rainfall_mm']:
        df_future[f'{col}_change'] = df_future[col].diff().fillna(0)
        df_future[f'{col}_pct_change'] = df_future[col].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    
    # Fill any missing features
    for col in feature_cols:
        if col not in df_future.columns:
            df_future[col] = 0
    
    # Ensure correct order and no NaN
    X = df_future[feature_cols].fillna(0)
    
    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    
    df_future['bloom_prediction'] = predictions
    df_future['bloom_probability'] = probabilities
    df_future['scenario'] = scenario
    
    # Risk classification
    df_future['risk_level'] = pd.cut(
        df_future['bloom_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    return df_future[['date', 'month', 'ndvi', 'max_temp', 'rainfall_mm',
                      'scenario', 'bloom_prediction', 'bloom_probability', 'risk_level']]

def save_forecasts(forecasts, output_path='outputs/predictions/bloom_forecast_enhanced.csv'):
    """Save forecast results."""
    import os
    os.makedirs('outputs/predictions', exist_ok=True)
    
    forecasts.to_csv(output_path, index=False)
    print(f"\n✅ Saved forecasts to {output_path}")

def print_forecast_summary(forecasts):
    """Print summary of forecasts."""
    
    print("\n" + "="*60)
    print("FORECAST SUMMARY (SEASONAL-AWARE)")
    print("="*60)
    
    for scenario in forecasts['scenario'].unique():
        df_scenario = forecasts[forecasts['scenario'] == scenario]
        bloom_count = df_scenario['bloom_prediction'].sum()
        high_risk_count = (df_scenario['risk_level'] == 'High').sum()
        medium_risk_count = (df_scenario['risk_level'] == 'Medium').sum()
        
        print(f"\n{scenario.upper()} Scenario:")
        print(f"  Forecast period: {df_scenario['date'].min().date()} to {df_scenario['date'].max().date()}")
        print(f"  Predicted blooms: {bloom_count}/{len(df_scenario)}")
        print(f"  High risk periods: {high_risk_count}")
        print(f"  Medium risk periods: {medium_risk_count}")
        print(f"  Average bloom probability: {df_scenario['bloom_probability'].mean():.2%}")
        
        # Show high-risk dates
        high_risk = df_scenario[df_scenario['risk_level'] == 'High']
        if len(high_risk) > 0:
            print(f"\n  ⚠️  HIGH-RISK DATES:")
            for _, row in high_risk.iterrows():
                print(f"    - {row['date'].strftime('%Y-%m-%d')} ({row['date'].strftime('%b')}) | " +
                      f"Probability: {row['bloom_probability']:.1%} | " +
                      f"NDVI: {row['ndvi']:.3f} | Temp: {row['max_temp']:.1f}°C")
        
        # Show medium-risk dates
        medium_risk = df_scenario[df_scenario['risk_level'] == 'Medium']
        if len(medium_risk) > 0:
            print(f"\n  ⚡ MEDIUM-RISK DATES:")
            for _, row in medium_risk.head(5).iterrows():
                print(f"    - {row['date'].strftime('%Y-%m-%d')} ({row['date'].strftime('%b')}) | " +
                      f"Probability: {row['bloom_probability']:.1%}")

def create_forecast_visualization(forecasts):
    """Create visualization of forecasts."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Bloom probability over time
    for scenario in forecasts['scenario'].unique():
        df_scenario = forecasts[forecasts['scenario'] == scenario]
        axes[0].plot(df_scenario['date'], df_scenario['bloom_probability'], 
                     marker='o', label=scenario.capitalize(), linewidth=2, markersize=6)
    
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold')
    axes[0].axhline(y=0.3, color='orange', linestyle=':', alpha=0.5, label='Medium Risk')
    axes[0].set_xlabel('Date', fontsize=12)
    axes[0].set_ylabel('Bloom Probability', fontsize=12)
    axes[0].set_title('Bloom Probability Forecast - All Scenarios (Seasonal-Aware)', 
                      fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Plot 2: Environmental variables (Normal scenario only)
    normal = forecasts[forecasts['scenario'] == 'normal']
    ax2_twin = axes[1].twinx()
    ax2_twin2 = axes[1].twinx()
    ax2_twin2.spines['right'].set_position(('outward', 60))
    
    l1 = axes[1].plot(normal['date'], normal['ndvi'], 'g-', marker='o', label='NDVI', linewidth=2)
    l2 = ax2_twin.plot(normal['date'], normal['max_temp'], 'r-', marker='s', label='Temperature', linewidth=2)
    l3 = ax2_twin2.plot(normal['date'], normal['rainfall_mm'], 'b-', marker='^', label='Rainfall', linewidth=2)
    
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].set_ylabel('NDVI', fontsize=12, color='g')
    ax2_twin.set_ylabel('Temperature (°C)', fontsize=12, color='r')
    ax2_twin2.set_ylabel('Rainfall (mm)', fontsize=12, color='b')
    axes[1].set_title('Predicted Environmental Variables - Normal Scenario', 
                      fontsize=14, fontweight='bold')
    
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    axes[1].legend(lns, labs, loc='upper left')
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Monthly bloom risk heatmap
    risk_by_month = forecasts.groupby(['scenario', 'month'])['bloom_probability'].mean().unstack(fill_value=0)
    sns.heatmap(risk_by_month, annot=True, fmt='.1%', cmap='RdYlGn_r', ax=axes[2], 
                cbar_kws={'label': 'Avg Bloom Probability'})
    axes[2].set_xlabel('Month', fontsize=12)
    axes[2].set_ylabel('Scenario', fontsize=12)
    axes[2].set_title('Average Bloom Risk by Month', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/forecast_enhanced.png', dpi=150)
    print(f"\n✅ Saved enhanced forecast visualization")
    plt.close()

def main():
    print("="*60)
    print("GENERATING ENHANCED BLOOM FORECASTS")
    print("(With Seasonal Awareness)")
    print("="*60)
    
    # Load model
    print("\n[Step 1] Loading trained model...")
    model, feature_cols = load_best_model('xgboost')
    
    # Load historical data
    print("\n[Step 2] Loading historical data...")
    df_historical = load_historical_data()
    last_date = df_historical['date'].max()
    print(f"   Last known date: {last_date.date()}")
    
    # Create future dates (1 year ahead)
    print("\n[Step 3] Creating future date range...")
    future_periods = 24  # 1 year (24 * 16 days ≈ 384 days)
    future_dates = create_future_dates(last_date, periods=future_periods)
    print(f"   Forecasting {future_periods} periods (~{future_periods * 16} days)")
    print(f"   Until: {future_dates['date'].max().date()}")
    
    # Generate forecasts for all scenarios
    print("\n[Step 4] Generating forecasts...")
    all_forecasts = []
    
    for scenario in ['normal', 'wet', 'dry']:
        forecast = generate_realistic_forecast(
            df_historical, future_dates, model, feature_cols, scenario
        )
        all_forecasts.append(forecast)
    
    forecasts = pd.concat(all_forecasts, ignore_index=True)
    
    # Save results
    print("\n[Step 5] Saving results...")
    save_forecasts(forecasts)
    
    # Create visualization
    print("\n[Step 6] Creating visualizations...")
    create_forecast_visualization(forecasts)
    
    # Print summary
    print_forecast_summary(forecasts)
    
    print("\n" + "="*60)
    print("✅ ENHANCED FORECASTING COMPLETE!")
    print("   Results saved to outputs/predictions/bloom_forecast_enhanced.csv")
    print("="*60)

if __name__ == "__main__":
    main()