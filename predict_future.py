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

def create_future_dates(last_date, periods=12):
    """Generate future dates (16-day intervals)."""
    future_dates = []
    current_date = last_date + timedelta(days=16)
    
    for i in range(periods):
        future_dates.append(current_date)
        current_date += timedelta(days=16)
    
    return pd.DataFrame({'date': future_dates})

def predict_with_confidence(model, X, feature_cols):
    """Make predictions with confidence intervals."""
    
    # Ensure features are in correct order
    X = X[feature_cols]
    
    # Get predictions and probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]  # Probability of bloom
    
    return predictions, probabilities

def generate_forecast_scenarios(df_historical, future_dates, model, feature_cols, scenarios=['normal', 'wet', 'dry']):
    """Generate forecasts under different weather scenarios."""
    
    all_forecasts = []
    
    for scenario in scenarios:
        print(f"\n   Generating {scenario.upper()} scenario...")
        
        df_future = future_dates.copy()
        
        # Get last known values
        last_row = df_historical.iloc[-1]
        
        # Scenario adjustments
        if scenario == 'wet':
            rainfall_multiplier = 1.5
            temp_adjustment = -2
            ndvi_multiplier = 1.1
        elif scenario == 'dry':
            rainfall_multiplier = 0.5
            temp_adjustment = 2
            ndvi_multiplier = 0.9
        else:  # normal
            rainfall_multiplier = 1.0
            temp_adjustment = 0
            ndvi_multiplier = 1.0
        
        # Initialize with last known values
        df_future['ndvi'] = last_row['ndvi'] * ndvi_multiplier
        df_future['max_temp'] = last_row['max_temp'] + temp_adjustment
        df_future['rainfall_mm'] = last_row['rainfall_mm'] * rainfall_multiplier
        
        # Add temporal features
        df_future['month'] = df_future['date'].dt.month
        df_future['year'] = df_future['date'].dt.year
        df_future['quarter'] = df_future['date'].dt.quarter
        df_future['day_of_year'] = df_future['date'].dt.dayofyear
        
        # Cyclical encoding
        df_future['month_sin'] = np.sin(2 * np.pi * df_future['month'] / 12)
        df_future['month_cos'] = np.cos(2 * np.pi * df_future['month'] / 12)
        df_future['day_sin'] = np.sin(2 * np.pi * df_future['day_of_year'] / 365)
        df_future['day_cos'] = np.cos(2 * np.pi * df_future['day_of_year'] / 365)
        
        # Create lag features using last historical values
        for col in ['ndvi', 'max_temp', 'rainfall_mm']:
            for lag in [1, 2, 3, 4]:
                if lag == 1:
                    df_future[f'{col}_lag{lag}'] = last_row[col]
                else:
                    df_future[f'{col}_lag{lag}'] = last_row[f'{col}_lag{lag-1}']
        
        # Rolling features (use historical averages)
        for col in ['ndvi', 'max_temp', 'rainfall_mm']:
            for window in [2, 4, 8]:
                df_future[f'{col}_roll_mean_{window}'] = df_historical[col].tail(window).mean()
                df_future[f'{col}_roll_std_{window}'] = df_historical[col].tail(window).std()
        
        # Interaction features
        df_future['temp_ndvi_interaction'] = df_future['max_temp'] * df_future['ndvi']
        df_future['temp_rainfall_interaction'] = df_future['max_temp'] * df_future['rainfall_mm']
        df_future['ndvi_rainfall_interaction'] = df_future['ndvi'] * df_future['rainfall_mm']
        df_future['cumulative_rain_2periods'] = df_future['rainfall_mm'] * 2
        df_future['cumulative_rain_4periods'] = df_future['rainfall_mm'] * 4
        
        # Rate of change (use last historical change)
        for col in ['ndvi', 'max_temp', 'rainfall_mm']:
            df_future[f'{col}_change'] = last_row[f'{col}_change']
            df_future[f'{col}_pct_change'] = last_row[f'{col}_pct_change']
        
        # Fill any missing features with 0
        for col in feature_cols:
            if col not in df_future.columns:
                df_future[col] = 0
        
        # Make predictions
        predictions, probabilities = predict_with_confidence(model, df_future, feature_cols)
        
        df_future['bloom_prediction'] = predictions
        df_future['bloom_probability'] = probabilities
        df_future['scenario'] = scenario
        
        # Risk classification
        df_future['risk_level'] = pd.cut(
            df_future['bloom_probability'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        all_forecasts.append(df_future[['date', 'scenario', 'bloom_prediction', 
                                         'bloom_probability', 'risk_level']])
    
    return pd.concat(all_forecasts, ignore_index=True)

def save_forecasts(forecasts, output_path='outputs/predictions/bloom_forecast.csv'):
    """Save forecast results."""
    import os
    os.makedirs('outputs/predictions', exist_ok=True)
    
    forecasts.to_csv(output_path, index=False)
    print(f"\n✅ Saved forecasts to {output_path}")

def print_forecast_summary(forecasts):
    """Print summary of forecasts."""
    
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    
    for scenario in forecasts['scenario'].unique():
        df_scenario = forecasts[forecasts['scenario'] == scenario]
        bloom_count = df_scenario['bloom_prediction'].sum()
        high_risk_count = (df_scenario['risk_level'] == 'High').sum()
        
        print(f"\n{scenario.upper()} Scenario:")
        print(f"  Forecast period: {df_scenario['date'].min().date()} to {df_scenario['date'].max().date()}")
        print(f"  Predicted blooms: {bloom_count}/{len(df_scenario)}")
        print(f"  High risk periods: {high_risk_count}")
        print(f"  Average bloom probability: {df_scenario['bloom_probability'].mean():.2%}")
        
        # Show high-risk dates
        high_risk_dates = df_scenario[df_scenario['risk_level'] == 'High']['date']
        if len(high_risk_dates) > 0:
            print(f"  High-risk dates:")
            for date in high_risk_dates:
                prob = df_scenario[df_scenario['date'] == date]['bloom_probability'].values[0]
                print(f"    - {date.date()} (probability: {prob:.1%})")

def create_forecast_visualization(forecasts):
    """Create visualization of forecasts."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Bloom probability over time for all scenarios
    for scenario in forecasts['scenario'].unique():
        df_scenario = forecasts[forecasts['scenario'] == scenario]
        axes[0].plot(df_scenario['date'], df_scenario['bloom_probability'], 
                     marker='o', label=scenario.capitalize(), linewidth=2)
    
    axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Decision Threshold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Bloom Probability')
    axes[0].set_title('Bloom Probability Forecast - All Scenarios', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 1)
    
    # Plot 2: Risk level distribution
    risk_counts = forecasts.groupby(['scenario', 'risk_level']).size().unstack(fill_value=0)
    risk_counts.plot(kind='bar', stacked=True, ax=axes[1], 
                     color=['green', 'orange', 'red'])
    axes[1].set_xlabel('Scenario')
    axes[1].set_ylabel('Number of Periods')
    axes[1].set_title('Risk Level Distribution by Scenario', fontsize=14, fontweight='bold')
    axes[1].legend(title='Risk Level')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/forecast_summary.png', dpi=150)
    print(f"\n✅ Saved forecast visualization")
    plt.close()

def main():
    print("="*60)
    print("GENERATING BLOOM FORECASTS")
    print("="*60)
    
    # Load model
    print("\n[Step 1] Loading trained model...")
    model, feature_cols = load_best_model('xgboost')  # Use best model
    
    # Load historical data
    print("\n[Step 2] Loading historical data...")
    df_historical = load_historical_data()
    last_date = df_historical['date'].max()
    print(f"   Last known date: {last_date.date()}")
    
    # Create future dates
    print("\n[Step 3] Creating future date range...")
    future_periods = 12  # ~6 months ahead (12 * 16 days)
    future_dates = create_future_dates(last_date, periods=future_periods)
    print(f"   Forecasting {future_periods} periods ({future_periods * 16} days)")
    print(f"   Until: {future_dates['date'].max().date()}")
    
    # Generate forecasts
    print("\n[Step 4] Generating forecasts for multiple scenarios...")
    forecasts = generate_forecast_scenarios(
        df_historical, 
        future_dates, 
        model, 
        feature_cols,
        scenarios=['normal', 'wet', 'dry']
    )
    
    # Save results
    print("\n[Step 5] Saving results...")
    save_forecasts(forecasts)
    
    # Create visualization
    print("\n[Step 6] Creating visualizations...")
    create_forecast_visualization(forecasts)
    
    # Print summary
    print_forecast_summary(forecasts)
    
    print("\n" + "="*60)
    print("✅ FORECASTING COMPLETE!")
    print("   Next step: python src/05_dashboard.py")
    print("="*60)

if __name__ == "__main__":
    main()