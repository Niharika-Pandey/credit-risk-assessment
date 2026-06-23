# End-to-End Credit Risk Assessment System

This repository contains a professional, production-grade Credit Risk Assessment machine learning pipeline and underwriting dashboard. Built with a risk analytics, this project predicts loan applicant default probability using the UCI German Credit dataset.


## Business Case & Objective

In retail banking, extending credit is a trade-off between **interest revenue generation** and **credit losses (defaults)**. 
* A **False Positive** (approving a borrower who defaults) costs the bank the outstanding principal of the loan (high cost).
* A **False Negative** (rejecting a creditworthy borrower) costs the bank the potential interest revenue (opportunity cost).

This project implements a modular ML pipeline to predict default risk (binary classification) and provides an interactive underwriting tool to support automated credit decisions, optimizing this risk-reward frontier.

---

## Project Structure

```
credit-risk-assessment/
│
├── data/
│   ├── raw/                  # Downloaded raw UCI dataset
│   └── processed/            # Cleaned, engineered, and scaled data
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset downloader & categorical decoder
│   ├── preprocessing.py      # Outlier capping, Standard Scaling, and One-Hot encoding
│   ├── feature_engineering.py# Financial ratios, age groups, and risk indices
│   ├── eda.py                # Visualizations (class distribution, default rates, etc.)
│   ├── models.py             # GridSearchCV parameter tuning & Stratified CV
│   ├── evaluation.py         # Performance evaluations, ROC & Confusion Matrices
│   ├── utils.py              # Visual themes and logging configuration
│   └── app.py                # Streamlit decisioning portal
│
├── models/
│   ├── *.pkl                 # Serialized model, preprocessor, and metadata files
│   └── pipeline_metadata.pkl # Feature lists and encoder metadata
│
├── outputs/                  # High-quality visual assets for presentations
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── age_distribution.png
│   ├── loan_amount_distribution.png
│   ├── default_vs_non_default.png
│   ├── roc_curve.png
│   ├── confusion_matrix_*.png
│   ├── feature_importance_*.png
│   └── model_comparison.json
│
├── requirements.txt          # Package dependencies
├── main.py                   # Master orchestration script
└── README.md                 # Project documentation
```

---

## Dataset Information

The model is trained on the **UCI German Credit Dataset** (1,000 cases).
* **Target Variable (`class`)**: 
  * `0` - Creditworthy (Low Risk / Good)
  * `1` - Default (High Risk / Bad)
* **Categorical Codes Decoder**: Raw dataset categorical features are encoded (e.g. `A11`, `A34`). Our pipeline programmatically maps these to descriptive text labels (e.g., `A11` $\rightarrow$ `< 0 DM`, `A34` $\rightarrow$ `Critical account / Other credits existing`) to enable intuitive visual analysis and production-ready inputs.

---

## Methodology & Pipeline Design

### 1. Data Quality & Cleaning
* **Outlier Capping**: Handled extreme outliers in numerical columns (`credit_amount`, `duration`, `age`) using the **Interquartile Range (IQR)** method (capping at $Q3 + 1.5 \times IQR$). This is preferred in credit risk over row deletion to preserve valuable borrower histories.
* **Imbalance Treatment**: The dataset is imbalanced (70% Good, 30% Bad). We incorporated stratification during train-test splitting and utilized class-weight adjustments in modeling.

### 2. Feature Engineering (Domain Ratios)
* **Monthly Payment Burden**: $\frac{\text{Credit Amount}}{\text{Duration}}$ representing monthly repayment pressure.
* **Credit-to-Age Ratio**: $\frac{\text{Credit Amount}}{\text{Age}}$ representing exposure relative to life stage.
* **Liquidity Index**: Numerical score combining Checking and Savings balances (scale of 2 to 9, where 9 indicates high liquidity buffer).
* **Stability Index**: Numerical score combining Employment Duration and Residence Duration.
* **Age Binning**: Grouped age into risk-aligned categories (`Young`, `Adult`, `Middle-Aged`, `Senior`).

### 3. Model Tuning (GridSearchCV)
A Grid Search with **5-Fold Stratified Cross-Validation** was executed to tune the models on the **ROC-AUC** metric, ensuring the models optimize class separation. The models trained are:
* **Logistic Regression**: High interpretability, standard for credit card underwriting (e.g., Amex).
* **Decision Tree**: Rule-based baseline.
* **Random Forest**: Ensemble bagging, robust against variance.
* **XGBoost**: Gradient boosting, capturing non-linear feature interactions.

---

## Model Evaluation & Performance

The models were evaluated on an unseen test set (20% holdout). Results are summarized below:

| Model | Test Accuracy | Test Precision | Test Recall (Sensitivity) | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **78.5%** | **69.8%** | **50.0%** | **58.3%** | **0.8201** |
| **Logistic Regression** | 79.0% | 68.8% | 55.0% | 61.1% | 0.8125 |
| **XGBoost** | 73.5% | 55.4% | 60.0% | 57.6% | 0.7863 |
| **Decision Tree** | 69.5% | 49.3% | 56.7% | 52.7% | 0.7078 |

### Key Takeaway:
* **Random Forest** achieved the highest test **ROC-AUC (0.8201)**, indicating the strongest probability ranking capability.
* **Logistic Regression** also performed exceptionally well (**0.8125 ROC-AUC**), representing a highly stable and interpretable baseline that meets strict regulatory standards in risk management (e.g., Basel requirements).

---

## Business Insights & Policy Recommendations

Based on feature importances and exploratory analysis:

1. **Checking Account Balance is the Primary Risk Indicator**:
   * Borrowers with **no checking account** or **checking balances $\ge$ 200 DM** have a default rate of less than **12%**.
   * In contrast, borrowers with a checking balance **$< 0$ DM (overdrawn)** have a default rate exceeding **49%**.
   * *Recommendation*: Implement an automated system check that flags or auto-declines applicants with history of overdrawn checking accounts unless significant collateral is provided.

2. **Liquid Savings Mitigates Default Risk**:
   * Having savings $\ge 1,000$ DM reduces default rate to less than **10%**.
   * *Recommendation*: Establish a policy that offers lower interest rates (risk-based pricing) to customers holding substantial deposits with the bank, creating an incentive for deposit accumulation.

3. **Duration is directly correlated with risk**:
   * Longer loan durations (e.g., 36+ months) dramatically increase default risk due to exposure to macroeconomic shifts.
   * *Recommendation*: Limit long-duration personal loans for younger applicants (<25 years old) or cap their credit exposure at 5,000 DM.

---

## How to Run the Project

### Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### 1. Install Dependencies
In your command prompt or terminal, navigate to the project directory and run:
```bash
pip install -r requirements.txt
```

### 2. Run the Training Pipeline
Run the orchestrator script to download the dataset, perform cleaning, train the models, and save all outputs:
```bash
python main.py
```
This script saves:
* Cleaned and processed datasets in `data/`
* Model comparison report and visual plots (ROC, Confusion Matrix, etc.) in `outputs/`
* Trained model pickles (`.pkl`) in `models/`

### 3. Launch the Underwriting Dashboard
Run the Streamlit application to open the interactive web interface:
```bash
streamlit run src/app.py
```
This opens the portal in your browser where you can input applicant details, select the model, and see real-time credit approval decisions and risk advisory recommendations.
