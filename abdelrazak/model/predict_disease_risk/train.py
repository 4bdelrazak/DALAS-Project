"""
Respiratory Disease Risk Prediction - Two Model Approach
=========================================================

Model 1 (Long-term): Meteorological data only (1980-2023)
    - Analyzes long-term effects of climate variables on health
    - 43 years of data
    
Model 2 (Short-term): Meteorological + Pollutant data (2003-2023)
    - Modern analysis with stronger correlations
    - Includes air pollution data (PM2.5, PM10, NO2, O3, SO2, CO)
    - 20 years of data

Usage:
    python train.py

Output:
    - model_longterm.joblib  (1980-2023, meteorological only)
    - model_shortterm.joblib (2003-2023, meteorological + pollutants)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # DALAS-Project root
DATA_DIR = PROJECT_ROOT
OUTPUT_DIR = SCRIPT_DIR

# Target diseases
TARGETS = [
    'asthma_M', 'asthma_F',
    'copd_M', 'copd_F', 
    'lri_M', 'lri_F',
    'lung_cancer_M', 'lung_cancer_F'
]

TARGET_NAMES = {
    'asthma_M': 'Asthma (Male)',
    'asthma_F': 'Asthma (Female)',
    'copd_M': 'COPD (Male)',
    'copd_F': 'COPD (Female)',
    'lri_M': 'Lower Respiratory Infection (Male)',
    'lri_F': 'Lower Respiratory Infection (Female)',
    'lung_cancer_M': 'Lung Cancer (Male)',
    'lung_cancer_F': 'Lung Cancer (Female)'
}

# Feature categories
POLLUTANT_KEYWORDS = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co', 'tcco', 'tcno2', 'gtco3', 'tcso2', 'PC_Pol']
METEOROLOGICAL_KEYWORDS = ['temp', 'precip', 'humidity', 'wind', 'pressure', 'radiation', 'uv', 
                           'dewpoint', 'season', 'winter', 'summer', 'PC_Met', 'tp_', 'ssrd', 
                           't2m', 'd2m', 'sp_', 'msl', 'skt', 'u10', 'v10', 'i10fg', 'lsp', 'cp_']


def load_data():
    """Load and merge climate and health data."""
    print("📂 Loading data...")
    
    # Load climate features
    climate_path = DATA_DIR / "Climate_Data_Yearly_Final.csv"
    climate_df = pd.read_csv(climate_path)
    print(f"   Climate data: {climate_df.shape}")
    print(f"   Years available: {climate_df['year'].min()} - {climate_df['year'].max()}")
    
    # Load health outcomes
    health_path = DATA_DIR / "new_disease_data_set" / "GBD_data.csv"
    health_df = pd.read_csv(health_path)
    print(f"   Health data: {health_df.shape}")
    
    # Merge
    df = pd.merge(climate_df, health_df, on=['country_name', 'year'], how='inner')
    print(f"   Merged: {df.shape}")
    
    return df


def categorize_features(df):
    """Separate features into meteorological and pollutant categories."""
    exclude = ['country_name', 'year', 'month'] + TARGETS
    exclude += [c for c in df.columns if c.endswith('_lo_F') or c.endswith('_lo_M')]
    exclude += [c for c in df.columns if c.endswith('_up_F') or c.endswith('_up_M')]
    
    all_features = [c for c in df.columns if c not in exclude]
    all_features = [c for c in all_features if df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    # Categorize
    pollutant_features = []
    meteorological_features = []
    
    for feat in all_features:
        feat_lower = feat.lower()
        is_pollutant = any(kw in feat_lower for kw in POLLUTANT_KEYWORDS)
        
        if is_pollutant:
            pollutant_features.append(feat)
        else:
            meteorological_features.append(feat)
    
    return meteorological_features, pollutant_features, all_features


def train_model(df, features, model_name, year_range, test_split_year=None):
    """Train a model with given features and year range."""
    print(f"\n{'='*60}")
    print(f"🧠 Training: {model_name}")
    print(f"   Year range: {year_range[0]} - {year_range[1]}")
    print(f"   Features: {len(features)}")
    print("=" * 60)
    
    # Filter by year
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    print(f"   Data points after year filter: {len(df_filtered)}")
    
    # Prepare data
    model_df = df_filtered[['country_name', 'year'] + features + TARGETS].dropna()
    print(f"   Samples after dropping NaN: {len(model_df)}")
    
    if len(model_df) < 100:
        print(f"   ⚠️ Insufficient data for training!")
        return None
    
    X = model_df[features].values
    y = model_df[TARGETS].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split (temporal) - use last 20% of years for testing if not specified
    if test_split_year is None:
        test_split_year = year_range[0] + int((year_range[1] - year_range[0]) * 0.8)
    
    train_mask = model_df['year'] <= test_split_year
    test_mask = model_df['year'] > test_split_year
    
    X_train, X_test = X_scaled[train_mask], X_scaled[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    print(f"   Train: {len(X_train)} samples ({year_range[0]}-{test_split_year})")
    print(f"   Test: {len(X_test)} samples ({test_split_year+1}-{year_range[1]})")
    
    if len(X_test) == 0:
        print("   ⚠️ No test data available, using 80/20 random split")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
    
    # Train multiple models and pick best
    print("\n   Training models...")
    
    models_to_try = {
        'GradientBoosting': MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            n_jobs=-1
        ),
        'RandomForest': MultiOutputRegressor(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        ),
    }
    
    best_model = None
    best_r2 = -np.inf
    best_name = None
    
    for name, model in models_to_try.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test.flatten(), y_pred.flatten())
        print(f"      {name}: R² = {r2:.3f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name
    
    print(f"\n   ✓ Best model: {best_name}")
    
    # Detailed evaluation
    print("\n   📊 Performance by Disease:")
    y_pred = best_model.predict(X_test)
    
    results = {}
    for i, target in enumerate(TARGETS):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        results[target] = {'r2': r2, 'rmse': rmse, 'mae': mae}
        print(f"      {TARGET_NAMES[target]:40s} R²={r2:.3f}  RMSE={rmse:.1f}")
    
    overall_r2 = r2_score(y_test.flatten(), y_pred.flatten())
    print(f"\n      {'OVERALL':40s} R²={overall_r2:.3f}")
    
    # Package model data
    model_data = {
        'model': best_model,
        'model_type': best_name,
        'scaler': scaler,
        'features': features,
        'targets': TARGETS,
        'target_names': TARGET_NAMES,
        'metrics': results,
        'overall_r2': overall_r2,
        'year_range': year_range,
        'train_years': f"{year_range[0]}-{test_split_year}",
        'test_years': f"{test_split_year+1}-{year_range[1]}",
        'n_samples': len(model_df),
        'n_features': len(features)
    }
    
    return model_data


def save_model(model_data, filename):
    """Save model to file."""
    output_path = OUTPUT_DIR / filename
    joblib.dump(model_data, output_path)
    print(f"   💾 Saved: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("🫁 RESPIRATORY DISEASE RISK PREDICTION")
    print("   Two-Model Training Approach")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Categorize features
    met_features, pol_features, all_features = categorize_features(df)
    
    print(f"\n📋 Feature Categories:")
    print(f"   Meteorological features: {len(met_features)}")
    print(f"   Pollutant features: {len(pol_features)}")
    print(f"   Total features: {len(all_features)}")
    
    # ========================================
    # MODEL 1: Long-term (Meteorological only)
    # ========================================
    model1 = train_model(
        df=df,
        features=met_features,
        model_name="Long-term Model (Meteorological Only)",
        year_range=(1980, 2023),
        test_split_year=2018
    )
    
    if model1:
        model1['description'] = "Long-term analysis using only meteorological variables (1980-2023)"
        save_model(model1, "model_longterm.joblib")
    
    # ========================================
    # MODEL 2: Short-term (Met + Pollutants)
    # ========================================
    model2 = train_model(
        df=df,
        features=all_features,  # Both meteorological and pollutant
        model_name="Short-term Model (Meteorological + Pollutants)",
        year_range=(2003, 2023),
        test_split_year=2018
    )
    
    if model2:
        model2['description'] = "Short-term modern analysis with meteorological and pollutant data (2003-2023)"
        save_model(model2, "model_shortterm.joblib")
    
    # Also save combined as the main model for the app
    if model2:
        save_model(model2, "model.joblib")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    
    print("\n📦 Models created:")
    if model1:
        print(f"\n   1. model_longterm.joblib")
        print(f"      - Years: 1980-2023 (43 years)")
        print(f"      - Features: {model1['n_features']} meteorological variables")
        print(f"      - Overall R²: {model1['overall_r2']:.3f}")
    
    if model2:
        print(f"\n   2. model_shortterm.joblib")
        print(f"      - Years: 2003-2023 (20 years)")
        print(f"      - Features: {model2['n_features']} (meteorological + pollutants)")
        print(f"      - Overall R²: {model2['overall_r2']:.3f}")
    
    print("\n" + "=" * 60)
    
    return model1, model2


if __name__ == "__main__":
    model1, model2 = main()
