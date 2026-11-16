#!/usr/bin/env python3
"""
Clinical Modeling & TreeSHAP Interpretability Pipeline
Demonstrates OLS Regression, Random Forest, and XGBoost with TreeSHAP feature attribution.
Executes automatically on synthetic dummy data.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from generate_synthetic_sci_sample import generate_synthetic_data

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def run_pipeline():
    print("1. Generating 200 synthetic clinical records...")
    df = generate_synthetic_data(n_samples=200, seed=101)
    
    feature_cols = ["AInjAge", "AFScorDs", "ANCatDis", "Sex", "BRhspNbr", "BBPHQSDS"]
    X = df[feature_cols]
    y = df["is_deceased"]
    
    # --- OLS Logistic / Linear Benchmark ---
    print("\n2. Fitting OLS Regression Benchmark...")
    X_sm = sm.add_constant(X)
    ols_model = sm.OLS(y, X_sm).fit()
    print("OLS Summary Top Features by t-value:")
    print(ols_model.tvalues.sort_values(ascending=False))
    
    # --- Random Forest ---
    print("\n3. Training Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    print("Random Forest Accuracy:", rf.score(X_test, y_test))
    
    # --- XGBoost ---
    if HAS_XGB:
        print("\n4. Training XGBoost Classifier...")
        xgb_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, eval_metric="logloss")
        xgb_model.fit(X_train, y_train)
        print("XGBoost Test Accuracy:", xgb_model.score(X_test, y_test))
        
        # --- TreeSHAP ---
        if HAS_SHAP:
            print("\n5. Computing TreeSHAP Values...")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X_test)
            print("TreeSHAP values shape:", np.array(shap_values).shape)
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Mean_SHAP_Impact": mean_abs_shap
            }).sort_values(by="Mean_SHAP_Impact", ascending=False)
            print("\nTreeSHAP Global Feature Importance:")
            print(importance_df.to_string(index=False))
        else:
            print("SHAP not installed; skipping TreeSHAP calculation.")
    else:
        print("XGBoost not installed; skipping XGBoost step.")

if __name__ == "__main__":
    run_pipeline()
