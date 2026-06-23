import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Credit Risk Decisioning Portal",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Amex/Consulting Aesthetic
st.markdown("""
<style>
    .main-title {
        font-family: 'Inter', sans-serif;
        color: #0F2942;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        color: #1A8C8C;
        font-weight: 500;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .card-approved {
        background-color: #E8F8F5;
        border-left: 6px solid #2ECC71;
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }
    .card-warning {
        background-color: #FEF9E7;
        border-left: 6px solid #F39C12;
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }
    .card-declined {
        background-color: #FDEDEC;
        border-left: 6px solid #D9534F;
        padding: 1.5rem;
        border-radius: 6px;
        margin-bottom: 1.5rem;
    }
    .card-title {
        font-weight: 700;
        font-size: 1.3rem;
        color: #2C3E50;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0F2942;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models and preprocessor
@st.cache_resource
def load_assets():
    models = {}
    model_names = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]
    for name in model_names:
        model_path = f"models/{name}_best.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                models[name] = pickle.load(f)
                
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
        
    with open("models/pipeline_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
        
    return models, preprocessor, metadata

try:
    models, preprocessor, metadata = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Error loading model assets. Make sure you run `python main.py` first to train the models. Error details: {e}")

# Sidebar - User Inputs
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-safe.png", width=80)
st.sidebar.header("Borrower Risk Profile")

# Group 1: Loan parameters
st.sidebar.subheader("Requested Credit Details")
credit_amount = st.sidebar.slider("Requested Credit Amount (DM)", 250, 20000, 3000, step=100)
duration = st.sidebar.slider("Loan Duration (Months)", 4, 72, 24)
purpose = st.sidebar.selectbox(
    "Purpose of Loan",
    ["New Car", "Used Car", "Furniture/Equipment", "Radio/Television", 
     "Domestic Appliances", "Repairs", "Education", "Retraining", "Business", "Others"]
)
installment_commitment = st.sidebar.selectbox(
    "Installment Rate (% of Disposable Income)",
    [1, 2, 3, 4], index=2
)

# Group 2: Financial Buffers
st.sidebar.subheader("Financial Standing")
checking_status = st.sidebar.selectbox(
    "Checking Account Status",
    ["< 0 DM", "0 <= ... < 200 DM", ">= 200 DM / salary assignments", "No checking account"],
    index=1
)
savings_status = st.sidebar.selectbox(
    "Savings Account Status",
    ["< 100 DM", "100 <= ... < 500 DM", "500 <= ... < 1000 DM", ">= 1000 DM", "Unknown / No savings account"],
    index=0
)
credit_history = st.sidebar.selectbox(
    "Credit History",
    ["No credits / All paid back duly", "All credits at this bank paid duly", 
     "Existing credits paid back duly till now", "Delay in past paying off", 
     "Critical account / Other credits existing"],
    index=2
)
existing_credits = st.sidebar.slider("Number of Existing Credits at this Bank", 1, 4, 1)

# Group 3: Demographic Stability
st.sidebar.subheader("Applicant Details")
age = st.sidebar.slider("Applicant Age (Years)", 18, 75, 35)
employment = st.sidebar.selectbox(
    "Employment Status (Present since)",
    ["Unemployed", "< 1 year", "1 <= ... < 4 years", "4 <= ... < 7 years", ">= 7 years"],
    index=2
)
personal_status = st.sidebar.selectbox(
    "Personal Status and Sex",
    ["Male : Divorced/Separated", "Female : Divorced/Separated/Married", 
     "Male : Single", "Male : Married/Widowed", "Female : Single"],
    index=2
)
housing = st.sidebar.selectbox(
    "Housing Type",
    ["Rent", "Own", "For Free"],
    index=1
)
residence_since = st.sidebar.selectbox(
    "Years at Present Residence",
    [1, 2, 3, 4], index=1
)
property_magnitude = st.sidebar.selectbox(
    "Property Magnitude",
    ["Real Estate", "Building Society Savings/Life Insurance", "Car or Other", "Unknown / No Property"],
    index=2
)
job = st.sidebar.selectbox(
    "Job Classification",
    ["Unemployed/Unskilled Non-resident", "Unskilled Resident", "Skilled Employee", "Management/Self-employed/Highly Qualified"],
    index=2
)
other_parties = st.sidebar.selectbox(
    "Other Debtors / Guarantors",
    ["None", "Co-applicant", "Guarantor"],
    index=0
)
other_payment_plans = st.sidebar.selectbox(
    "Other Installment Plans",
    ["Bank", "Stores", "None"],
    index=2
)
num_dependents = st.sidebar.selectbox(
    "Number of Dependents",
    [1, 2], index=0
)
own_telephone = st.sidebar.selectbox(
    "Telephone Registry",
    ["None", "Yes"],
    index=0
)
foreign_worker = st.sidebar.selectbox(
    "Foreign Worker Status",
    ["Yes", "No"],
    index=0
)

# Main Panel
st.markdown('<div class="main-title">Credit Risk Decisioning Portal</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-Powered Risk Engine & Credit Scoring System for Underwriting</div>', unsafe_allow_html=True)

if assets_loaded:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Model Configuration")
        model_name = st.selectbox(
            "Select Risk Assessment Model",
            ["RandomForest", "LogisticRegression", "XGBoost", "DecisionTree"]
        )
        
        # Build the input dataframe matching the exact schema (original features)
        input_data = pd.DataFrame({
            "checking_status": [checking_status],
            "duration": [duration],
            "credit_history": [credit_history],
            "purpose": [purpose],
            "credit_amount": [credit_amount],
            "savings_status": [savings_status],
            "employment": [employment],
            "installment_commitment": [installment_commitment],
            "personal_status": [personal_status],
            "other_parties": [other_parties],
            "residence_since": [residence_since],
            "property_magnitude": [property_magnitude],
            "age": [age],
            "other_payment_plans": [other_payment_plans],
            "housing": [housing],
            "existing_credits": [existing_credits],
            "job": [job],
            "num_dependents": [num_dependents],
            "own_telephone": [own_telephone],
            "foreign_worker": [foreign_worker]
        })
        
        # Apply the exact feature engineering steps used during training
        # We need to replicate the mappings and features engineered in feature_engineering.py
        input_eng = input_data.copy()
        input_eng["monthly_payment_burden"] = input_eng["credit_amount"] / (input_eng["duration"] + 1e-5)
        input_eng["credit_to_age_ratio"] = input_eng["credit_amount"] / (input_eng["age"] + 1e-5)
        
        # Age group cut
        bins = [0, 25, 45, 60, 100]
        labels = ["Young", "Adult", "Middle-Aged", "Senior"]
        input_eng["age_group"] = pd.cut(input_eng["age"], bins=bins, labels=labels)
        input_eng["age_group"] = input_eng["age_group"].astype(str)
        
        # Map ordinal scores
        checking_risk_map = {"< 0 DM": 1, "0 <= ... < 200 DM": 2, "No checking account": 3, ">= 200 DM / salary assignments": 4}
        savings_risk_map = {"< 100 DM": 1, "Unknown / No savings account": 2, "100 <= ... < 500 DM": 3, "500 <= ... < 1000 DM": 4, ">= 1000 DM": 5}
        employment_map = {"Unemployed": 1, "< 1 year": 2, "1 <= ... < 4 years": 3, "4 <= ... < 7 years": 4, ">= 7 years": 5}
        
        input_eng["checking_score"] = input_eng["checking_status"].map(checking_risk_map).fillna(2)
        input_eng["savings_score"] = input_eng["savings_status"].map(savings_risk_map).fillna(2)
        input_eng["employment_score"] = input_eng["employment"].map(employment_map).fillna(3)
        
        # Indexes
        input_eng["liquidity_index"] = input_eng["checking_score"] + input_eng["savings_score"]
        input_eng["stability_index"] = input_eng["employment_score"] + input_eng["residence_since"]
        
        # Preprocess features (impute & scale numerical, one-hot encode categorical)
        try:
            input_processed = preprocessor.transform(input_eng)
            
            # Predict
            model = models[model_name]
            prediction = model.predict(input_processed)[0]
            
            if hasattr(model, "predict_proba"):
                prob_default = model.predict_proba(input_processed)[0][1]
            else:
                raw_decision = model.decision_function(input_processed)[0]
                prob_default = 1 / (1 + np.exp(-raw_decision)) # sigmoid
                
            # Score (Standard credit score representation 300-850)
            # Invert default probability to get credit score: 300 + (1 - prob) * 550
            credit_score = int(300 + (1 - prob_default) * 550)
            
            st.subheader("Credit Decisioning Output")
            
            if prob_default < 0.30:
                st.markdown(f"""
                <div class="card-approved">
                    <div class="card-title">✔ APPLICATION APPROVED (Low Risk)</div>
                    <p>The applicant exhibits a low risk of default. Standard credit lines can be extended.</p>
                    <p>Default Probability: <b>{prob_default*100:.1f}%</b> | Estimated Credit Score: <b>{credit_score}</b></p>
                </div>
                """, unsafe_allow_html=True)
            elif prob_default < 0.50:
                st.markdown(f"""
                <div class="card-warning">
                    <div class="card-title">⚠ CONDITIONAL APPROVAL (Medium Risk)</div>
                    <p>The applicant is flagged as medium risk. Consider lower credit limits, higher interest rates, or requiring a guarantor.</p>
                    <p>Default Probability: <b>{prob_default*100:.1f}%</b> | Estimated Credit Score: <b>{credit_score}</b></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card-declined">
                    <div class="card-title">✖ APPLICATION DECLINED (High Risk)</div>
                    <p>The risk of default exceeds the bank's risk appetite. Rejecting this application is recommended to prevent credit loss.</p>
                    <p>Default Probability: <b>{prob_default*100:.1f}%</b> | Estimated Credit Score: <b>{credit_score}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as ex:
            st.error(f"Error preprocessing or predicting: {ex}")
            
    with col2:
        st.subheader("Risk Analytics & Score Breakdown")
        
        # 1. Gauge chart for Default Probability
        fig, ax = plt.subplots(figsize=(6, 2.5))
        # Plot horizontal bar for probability
        bar_colors = ["#2ECC71", "#F39C12", "#D9534F"]
        color = bar_colors[0] if prob_default < 0.30 else (bar_colors[1] if prob_default < 0.50 else bar_colors[2])
        
        ax.barh(["Default Risk"], [prob_default], color=color, height=0.4, edgecolor="#333333", linewidth=1.2)
        ax.barh(["Default Risk"], [1 - prob_default], left=[prob_default], color="#EAEAEA", height=0.4)
        
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.30, 0.50, 1.0])
        ax.set_xticklabels(["0% (Perfect)", "30% (Warning)", "50% (Critical)", "100% (Default)"])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.grid(False)
        
        # Add text label on the bar
        ax.text(prob_default/2, 0, f"{prob_default*100:.1f}%", ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        
        st.pyplot(fig)
        plt.close()
        
        # Key ratios display
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div style="text-align: center; background-color: #F8F9FA; padding: 10px; border-radius: 5px;">
                <div style="font-size: 0.9rem; color: #7F8C8D;">Debt Burden Ratio</div>
                <div class="metric-value">{installment_commitment}%</div>
                <div style="font-size: 0.8rem; color: #95A5A6;">Disposable Income</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div style="text-align: center; background-color: #F8F9FA; padding: 10px; border-radius: 5px;">
                <div style="font-size: 0.9rem; color: #7F8C8D;">Monthly Payment</div>
                <div class="metric-value">{int(credit_amount/duration)} DM</div>
                <div style="font-size: 0.8rem; color: #95A5A6;">Over {duration} Months</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div style="text-align: center; background-color: #F8F9FA; padding: 10px; border-radius: 5px;">
                <div style="font-size: 0.9rem; color: #7F8C8D;">Liquidity Index</div>
                <div class="metric-value">{int(input_eng['liquidity_index'].values[0])} / 9</div>
                <div style="font-size: 0.8rem; color: #95A5A6;">Savings + Checking</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Explanatory details
        st.markdown("### Risk Mitigation Recommendations")
        if prob_default >= 0.50:
            st.info("""
            **Credit Risk Advisory:** 
            - Decline this standard loan. 
            - If strategic interest exists, renegotiate terms: demand collateral (e.g. real estate), reduce the credit amount by at least 40%, or require a high-liquidity Co-applicant/Guarantor.
            """)
        elif prob_default >= 0.30:
            st.info("""
            **Credit Risk Advisory:** 
            - Approve with strict pricing adjustments: apply a risk premium (+2.5% to interest rates).
            - Restrict loan maturity (shorten duration) to reduce overall exposure time.
            """)
        else:
            st.info("""
            **Credit Risk Advisory:**
            - Standard approval. Excellent credit capacity. 
            - Upsell opportunities: offer credit cards or overdraft limits, given the solid checking account standing.
            """)

else:
    st.info("System is ready but missing model assets. Please run the model training step to generate saved pipeline files.")
