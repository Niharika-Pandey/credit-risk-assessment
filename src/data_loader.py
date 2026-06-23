import os
import pandas as pd
import requests
import logging

UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Mapping dictionaries to convert raw codes into human-readable categories
MAPPINGS = {
    "checking_status": {
        "A11": "< 0 DM",
        "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM / salary assignments",
        "A14": "No checking account"
    },
    "credit_history": {
        "A30": "No credits / All paid back duly",
        "A31": "All credits at this bank paid duly",
        "A32": "Existing credits paid back duly till now",
        "A33": "Delay in past paying off",
        "A34": "Critical account / Other credits existing"
    },
    "purpose": {
        "A40": "New Car",
        "A41": "Used Car",
        "A42": "Furniture/Equipment",
        "A43": "Radio/Television",
        "A44": "Domestic Appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others"
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "Unknown / No savings account"
    },
    "employment": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years",
        "A75": ">= 7 years"
    },
    "personal_status": {
        "A91": "Male : Divorced/Separated",
        "A92": "Female : Divorced/Separated/Married",
        "A93": "Male : Single",
        "A94": "Male : Married/Widowed",
        "A95": "Female : Single"
    },
    "other_parties": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor"
    },
    "property_magnitude": {
        "A121": "Real Estate",
        "A122": "Building Society Savings/Life Insurance",
        "A123": "Car or Other",
        "A124": "Unknown / No Property"
    },
    "other_payment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None"
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "For Free"
    },
    "job": {
        "A171": "Unemployed/Unskilled Non-resident",
        "A172": "Unskilled Resident",
        "A173": "Skilled Employee",
        "A174": "Management/Self-employed/Highly Qualified"
    },
    "own_telephone": {
        "A191": "None",
        "A192": "Yes"
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No"
    }
}

COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job", "num_dependents",
    "own_telephone", "foreign_worker", "class"
]

def download_dataset(raw_data_path="data/raw/german_credit_raw.csv"):
    """Downloads the German Credit dataset from UCI and saves it as CSV."""
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    
    if os.path.exists(raw_data_path):
        logging.info(f"Dataset already exists at {raw_data_path}. Skipping download.")
        return pd.read_csv(raw_data_path)
    
    logging.info(f"Downloading German Credit dataset from {UCI_DATA_URL}...")
    try:
        response = requests.get(UCI_DATA_URL, timeout=15)
        response.raise_for_status()
        
        # Parse data which is space separated
        lines = [line.strip().split() for line in response.text.strip().split("\n")]
        df = pd.DataFrame(lines, columns=COLUMNS)
        
        # Numeric columns parsing
        numeric_cols = ["duration", "credit_amount", "installment_commitment", 
                        "residence_since", "age", "existing_credits", "num_dependents", "class"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
            
        # Save raw data
        df.to_csv(raw_data_path, index=False)
        logging.info(f"Dataset successfully downloaded and saved to {raw_data_path}")
        return df
    except Exception as e:
        logging.error(f"Failed to download dataset: {e}")
        raise e

def load_and_map_data(raw_data_path="data/raw/german_credit_raw.csv"):
    """Loads raw data and maps codes to human-readable text."""
    if not os.path.exists(raw_data_path):
        df = download_dataset(raw_data_path)
    else:
        df = pd.read_csv(raw_data_path)
        
    logging.info("Applying descriptive mapping to categorical columns...")
    df_mapped = df.copy()
    
    # Apply category mappings
    for col, mapping in MAPPINGS.items():
        if col in df_mapped.columns:
            df_mapped[col] = df_mapped[col].map(mapping).fillna(df_mapped[col])
            
    # Map class target variable: 1 = Good (Creditworthy/0), 2 = Bad (Default/1)
    # The original dataset labels 1 as Good and 2 as Bad. We want standard binary format:
    df_mapped["class"] = df_mapped["class"].map({1: 0, 2: 1})
    
    logging.info("Data mapping completed successfully.")
    return df_mapped

if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging()
    download_dataset()
    df = load_and_map_data()
    print(df.head())
