#!/usr/bin/env python3
"""
Synthetic SCI Clinical Dataset Generator
Creates a 100-row dummy dataset strictly matching the Form I, Form II,
and Record Status feature schema for local testing of ML and TreeSHAP pipelines.
Contains ZERO patient or clinical data.
"""

import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=100, seed=42):
    np.random.seed(seed)
    
    uni_ids = [f"SYNTH_{i:04d}" for i in range(1, n_samples + 1)]
    
    # Form I Baseline
    age_injury = np.random.randint(18, 80, size=n_samples)
    fIM_score = np.random.uniform(18.0, 126.0, size=n_samples) # Motor/cognitive functional score
    neuro_cat = np.random.choice([1, 2, 3, 4, 5], size=n_samples) # Tetraplegia / Paraplegia levels
    sex = np.random.choice([1, 2], size=n_samples, p=[0.78, 0.22]) # 1=Male, 2=Female
    race = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.65, 0.25, 0.05, 0.05])
    returned_home = np.random.choice([0, 1], size=n_samples, p=[0.15, 0.85])
    
    # Form II Longitudinal Follow-up
    followup_year = np.random.choice([1, 5, 10, 15, 20], size=n_samples)
    rehospitalized = np.random.poisson(lam=0.4, size=n_samples)
    able_to_drive = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
    phq9_depression = np.random.randint(0, 27, size=n_samples) # PHQ-9 score
    
    # Record Status
    deceased = np.random.choice([0, 1], size=n_samples, p=[0.82, 0.18])
    status = ["Deceased" if d == 1 else "Eligible" for d in deceased]
    
    categories = ["Cardiovascular", "Respiratory", "Infection/Septicemia", "External/Unintentional", "Other/Unknown"]
    death_cat = [np.random.choice(categories) if d == 1 else "Alive" for d in deceased]
    
    df = pd.DataFrame({
        "UniID": uni_ids,
        "AInjAge": age_injury,
        "AFScorDs": fIM_score,
        "ANCatDis": neuro_cat,
        "Sex": sex,
        "Race": race,
        "APResDis": returned_home,
        "BYear": followup_year,
        "BRhspNbr": rehospitalized,
        "BMdTrnsp": able_to_drive,
        "BBPHQSDS": phq9_depression,
        "SStatCur": status,
        "is_deceased": deceased,
        "primary_death_category": death_cat
    })
    
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(100)
    out_path = "synthetic_sci_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} synthetic patient records -> {out_path}")
    print(df.head())
