import pandas as pd
import numpy as np
import logging

def map_ordinal_features(df):
    """Maps critical categorical variables to numerical ranks to calculate domain risk indexes."""
    df_mapped = df.copy()
    
    # Map checking account status to order of liquidity/safety
    # < 0 DM is lowest liquidity/highest risk. No checking is typically low risk (or unknown) in German dataset.
    checking_risk_map = {
        "< 0 DM": 1,
        "0 <= ... < 200 DM": 2,
        "No checking account": 3,
        ">= 200 DM / salary assignments": 4
    }
    df_mapped["checking_score"] = df_mapped["checking_status"].map(checking_risk_map).fillna(2)
    
    # Map savings account status to order of magnitude
    savings_risk_map = {
        "< 100 DM": 1,
        "Unknown / No savings account": 2,
        "100 <= ... < 500 DM": 3,
        "500 <= ... < 1000 DM": 4,
        ">= 1000 DM": 5
    }
    df_mapped["savings_score"] = df_mapped["savings_status"].map(savings_risk_map).fillna(2)
    
    # Map employment duration to duration rank
    employment_map = {
        "Unemployed": 1,
        "< 1 year": 2,
        "1 <= ... < 4 years": 3,
        "4 <= ... < 7 years": 4,
        ">= 7 years": 5
    }
    df_mapped["employment_score"] = df_mapped["employment"].map(employment_map).fillna(3)
    
    return df_mapped

def engineer_features(df):
    """Engineers domain-specific financial risk features from the mapped credit dataset."""
    logging.info("Starting feature engineering...")
    df_eng = df.copy()
    
    # 1. Monthly credit burden (Credit Amount divided by Duration)
    # Higher values indicate high monthly repayment pressure
    df_eng["monthly_payment_burden"] = df_eng["credit_amount"] / (df_eng["duration"] + 1e-5)
    
    # 2. Credit exposure to Age ratio
    # Captures exposure risk relative to the borrower's life stage
    df_eng["credit_to_age_ratio"] = df_eng["credit_amount"] / (df_eng["age"] + 1e-5)
    
    # 3. Age Binning (Categorical age groups based on risk profiles)
    # Younger applicants (<25) and Seniors (>60) may exhibit different default behaviors
    bins = [0, 25, 45, 60, 100]
    labels = ["Young", "Adult", "Middle-Aged", "Senior"]
    df_eng["age_group"] = pd.cut(df_eng["age"], bins=bins, labels=labels)
    df_eng["age_group"] = df_eng["age_group"].astype(str) # Convert category to string for One-Hot encoding
    
    # 4. Map ordinals to compute score indexes
    df_eng = map_ordinal_features(df_eng)
    
    # 5. Liquidity Index (Checking Account Score + Savings Account Score)
    # Measures the overall immediate financial buffer of the applicant
    df_eng["liquidity_index"] = df_eng["checking_score"] + df_eng["savings_score"]
    
    # 6. Stability Index (Employment Score + Residence Duration Score)
    # Captures stability of income source and housing
    df_eng["stability_index"] = df_eng["employment_score"] + df_eng["residence_since"]
    
    # Drop intermediate score columns if desired, but keeping them as features can boost model performance
    
    logging.info(f"Feature engineering completed. Shape changed from {df.shape} to {df_eng.shape}")
    return df_eng

if __name__ == "__main__":
    from src.utils import setup_logging
    from src.data_loader import load_and_map_data
    setup_logging()
    df = load_and_map_data()
    df_eng = engineer_features(df)
    print(df_eng.head())
