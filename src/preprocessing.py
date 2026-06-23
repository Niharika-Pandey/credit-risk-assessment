import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def remove_duplicates(df):
    """Checks and removes duplicate rows from the dataset."""
    initial_shape = df.shape
    df_cleaned = df.drop_duplicates()
    final_shape = df_cleaned.shape
    diff = initial_shape[0] - final_shape[0]
    if diff > 0:
        logging.info(f"Removed {diff} duplicate rows.")
    else:
        logging.info("No duplicate rows found.")
    return df_cleaned

def handle_missing_values(df):
    """Imputes missing values. 
    German Credit dataset doesn't have missing values, but this functions acts as a safety guard.
    """
    logging.info("Checking for missing values...")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logging.warning(f"Missing values found: {missing_counts[missing_counts > 0]}")
        # Standard filling for demonstration
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
    else:
        logging.info("No missing values detected.")
    return df

def cap_outliers(df, columns=["credit_amount", "duration"]):
    """Caps outliers in numeric columns using the IQR (Interquartile Range) method.
    This is preferred in risk modeling over deleting rows, to preserve credit history records.
    """
    logging.info("Analyzing and treating outliers...")
    df_capped = df.copy()
    for col in columns:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count how many will be capped
        outliers_lower = (df_capped[col] < lower_bound).sum()
        outliers_upper = (df_capped[col] > upper_bound).sum()
        if outliers_lower > 0 or outliers_upper > 0:
            logging.info(f"Capping {outliers_lower} lower and {outliers_upper} upper outliers in '{col}' at bounds [{lower_bound:.1f}, {upper_bound:.1f}]")
            
        df_capped[col] = np.clip(df_capped[col], lower_bound, upper_bound)
    return df_capped

def create_preprocessing_pipeline(numeric_cols, categorical_cols):
    """Creates a scikit-learn ColumnTransformer for preprocessing numeric and categorical columns."""
    numeric_transformer = ColumnTransformer(
        transformers=[
            ("imputer", SimpleImputer(strategy="median"), numeric_cols),
            ("scaler", StandardScaler(), numeric_cols)
        ]
    )
    
    # We create a pipeline that handles missing values and then applies standard scaling/one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    return preprocessor

def preprocess_and_split(df, target_col="class", test_size=0.2, random_state=42, save_path="models/preprocessor.pkl"):
    """Performs data preprocessing, splits into train/test, fits pipeline, and exports encoders/scalers."""
    # 1. Clean data
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = cap_outliers(df, columns=["credit_amount", "duration", "age"])
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    
    logging.info(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
    logging.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    
    # 2. Split into train and test sets (stratified to maintain default ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # 3. Fit preprocessing pipeline on training data only (to prevent data leakage)
    preprocessor = create_preprocessing_pipeline(numeric_cols, categorical_cols)
    logging.info("Fitting preprocessing pipeline on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get feature names after one-hot encoding for modeling interpretation
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_cols).tolist()
    feature_names = numeric_cols + encoded_cat_cols
    
    # Save preprocessor object
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(preprocessor, f)
    logging.info(f"Preprocessor successfully fitted and saved to {save_path}")
    
    # Also save column lists and feature names for later use
    meta_path = "models/pipeline_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "feature_names": feature_names
        }, f)
    logging.info(f"Pipeline metadata saved to {meta_path}")
    
    # Return as dataframes/arrays along with features list
    return X_train_processed, X_test_processed, y_train, y_test, feature_names
