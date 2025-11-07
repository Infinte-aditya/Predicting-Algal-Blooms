import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_ml_data():
    """Load feature-engineered dataset."""
    df = pd.read_csv('data/processed/ml_features.csv', parse_dates=['date'])
    print(f"Loaded {len(df)} records")
    return df

def prepare_train_test_split(df, target='bloom_next_1', test_size=0.2):
    """Split data into train/test with temporal awareness."""
    
    # Define features (exclude date, geometry, targets, and original labels)
    exclude_cols = ['date', 'geometry', 'bloom', 'bloom_next_1', 'bloom_next_2', 
                    'bloom_next_4', 'risk_next_1', 'risk_next_2', 'risk_score']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df[target].copy()
    
    # Handle any remaining NaN - CRITICAL FIX
    print(f"\nChecking for NaN values...")
    nan_counts = X.isna().sum()
    if nan_counts.sum() > 0:
        print(f"   Found {nan_counts.sum()} NaN values across {(nan_counts > 0).sum()} columns")
        print(f"   Filling NaN with column means...")
    
    # Fill NaN with mean for numeric columns
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(0)
    
    # Double-check: replace any remaining NaN with 0
    X = X.fillna(0)
    
    # Verify no NaN remains
    if X.isna().any().any():
        print("   WARNING: Still have NaN values. Replacing all with 0.")
        X = X.fillna(0)
    else:
        print(f"   ✓ No NaN values in features")
    
    # Check for infinite values
    if np.isinf(X.values).any():
        print(f"   Found infinite values. Replacing with 0...")
        X = X.replace([np.inf, -np.inf], 0)
    
    # Time-based split (last 20% as test)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain set: {len(X_train)} samples ({y_train.sum()} blooms)")
    print(f"Test set: {len(X_test)} samples ({y_test.sum()} blooms)")
    
    return X_train, X_test, y_train, y_test, feature_cols

def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    print("\n[1/5] Training Random Forest...")
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle imbalanced data
    )
    
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    print("[2/5] Training XGBoost...")
    
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    return xgb_model

def train_lightgbm(X_train, y_train):
    """Train LightGBM classifier."""
    print("[3/5] Training LightGBM...")
    
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    lgb_model.fit(X_train, y_train)
    return lgb_model

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting classifier."""
    print("[4/5] Training Gradient Boosting...")
    
    # Use HistGradientBoostingClassifier which handles NaN natively
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    gb = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    
    gb.fit(X_train, y_train)
    return gb

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression (baseline)."""
    print("[5/5] Training Logistic Regression (baseline)...")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    
    lr.fit(X_train_scaled, y_train)
    return lr, scaler

def evaluate_model(model, X_test, y_test, model_name, scaler=None):
    """Evaluate model performance."""
    
    # Scale if needed (for Logistic Regression)
    X_test_eval = scaler.transform(X_test) if scaler else X_test
    
    # Predictions
    y_pred = model.predict(X_test_eval)
    y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
    
    # Metrics
    accuracy = (y_pred == y_test).mean()
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*50}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Bloom', 'Bloom']))
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

def plot_feature_importance(model, feature_cols, model_name, top_n=20):
    """Plot top N most important features."""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        return  # Skip if model doesn't have feature importances
    
    # Create dataframe
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feat_imp, x='importance', y='feature', palette='viridis')
    plt.title(f'{model_name} - Top {top_n} Features', fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    os.makedirs('outputs/visualizations', exist_ok=True)
    plt.savefig(f'outputs/visualizations/{model_name}_feature_importance.png', dpi=150)
    print(f"   Saved feature importance plot")
    plt.close()
    
    return feat_imp

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot confusion matrix."""
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Bloom', 'Bloom'],
                yticklabels=['No Bloom', 'Bloom'])
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(f'outputs/visualizations/{model_name}_confusion_matrix.png', dpi=150)
    print(f"   Saved confusion matrix plot")
    plt.close()

def plot_roc_curves(results, y_test):
    """Plot ROC curves for all models."""
    
    plt.figure(figsize=(10, 8))
    
    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {result['roc_auc']:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('outputs/visualizations/all_models_roc_curves.png', dpi=150)
    print(f"\n✅ Saved combined ROC curves plot")
    plt.close()

def save_models(models, feature_cols):
    """Save trained models."""
    
    os.makedirs('data/models', exist_ok=True)
    
    for name, model_data in models.items():
        model_path = f'data/models/{name}.joblib'
        joblib.dump(model_data, model_path)
        print(f"   Saved {name} to {model_path}")
    
    # Save feature list
    joblib.dump(feature_cols, 'data/models/feature_columns.joblib')
    print(f"   Saved feature list")

def main():
    print("="*60)
    print("TRAINING BLOOM PREDICTION MODELS")
    print("="*60)
    
    # Load data
    print("\n[Step 1] Loading data...")
    df = load_ml_data()
    
    # Prepare train/test split
    print("\n[Step 2] Preparing train/test split...")
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)
    
    # Train models
    print("\n[Step 3] Training models...")
    rf = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    lgb_model = train_lightgbm(X_train, y_train)
    gb = train_gradient_boosting(X_train, y_train)
    lr, scaler = train_logistic_regression(X_train, y_train)
    
    # Evaluate all models
    print("\n[Step 4] Evaluating models...")
    results = []
    
    results.append(evaluate_model(rf, X_test, y_test, 'Random Forest'))
    results.append(evaluate_model(xgb_model, X_test, y_test, 'XGBoost'))
    results.append(evaluate_model(lgb_model, X_test, y_test, 'LightGBM'))
    results.append(evaluate_model(gb, X_test, y_test, 'Gradient Boosting'))
    results.append(evaluate_model(lr, X_test, y_test, 'Logistic Regression', scaler))
    
    # Find best model
    best_result = max(results, key=lambda x: x['roc_auc'])
    print(f"\n{'='*60}")
    print(f"🏆 BEST MODEL: {best_result['model_name'].upper()}")
    print(f"   ROC-AUC: {best_result['roc_auc']:.3f}")
    print(f"   Accuracy: {best_result['accuracy']:.3f}")
    print(f"{'='*60}")
    
    # Visualizations
    print("\n[Step 5] Creating visualizations...")
    
    # Feature importance for tree-based models
    plot_feature_importance(xgb_model, feature_cols, 'XGBoost')
    plot_feature_importance(lgb_model, feature_cols, 'LightGBM')
    plot_feature_importance(rf, feature_cols, 'Random_Forest')
    
    # Confusion matrices
    for result in results:
        plot_confusion_matrix(y_test, result['y_pred'], result['model_name'])
    
    # Combined ROC curves
    plot_roc_curves(results, y_test)
    
    # Save models
    print("\n[Step 6] Saving models...")
    models = {
        'random_forest': rf,
        'xgboost': xgb_model,
        'lightgbm': lgb_model,
        'gradient_boosting': gb,
        'logistic_regression': {'model': lr, 'scaler': scaler}
    }
    save_models(models, feature_cols)
    
    # Save results summary
    results_df = pd.DataFrame([{
        'model': r['model_name'],
        'accuracy': r['accuracy'],
        'roc_auc': r['roc_auc']
    } for r in results]).sort_values('roc_auc', ascending=False)
    
    results_df.to_csv('outputs/model_comparison.csv', index=False)
    print(f"\n✅ Saved model comparison to outputs/model_comparison.csv")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("   Next step: python src/04_predict_future.py")
    print("="*60)

if __name__ == "__main__":
    main()