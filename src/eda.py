import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import get_palette

def generate_eda_visualizations(df, output_dir="outputs/"):
    """Generates and saves professional financial EDA plots to the outputs folder."""
    logging.info("Starting exploratory data analysis (EDA) visualization generation...")
    os.makedirs(output_dir, exist_ok=True)
    palette = get_palette()
    
    # 1. Class Distribution (Good vs Bad Loan Defaults)
    plt.figure(figsize=(6, 5))
    class_counts = df["class"].value_counts().sort_index()
    # Map 0 -> Creditworthy, 1 -> Default
    class_labels = ["Creditworthy\n(Low Risk)", "Default\n(High Risk)"]
    
    bars = plt.bar(class_labels, class_counts, color=palette["binary"], width=0.6, edgecolor="#333333", linewidth=1.2)
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        plt.text(
            bar.get_x() + bar.get_width()/2., height + 15,
            f"{int(height)}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, weight="bold"
        )
        
    plt.title("Loan Applicant Class Distribution (Imbalance Check)", pad=15)
    plt.ylabel("Number of Applicants")
    plt.ylim(0, max(class_counts) + 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "class_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Correlation Heatmap of Numeric Features
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    # Exclude the target variable for a pure feature correlation analysis
    features_corr = numeric_df.drop(columns=["class"], errors="ignore")
    corr_matrix = features_corr.corr()
    
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    plt.title("Correlation Matrix of Numeric Features", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    
    # 3. Income Distribution (Using Age as demographic proxy)
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df, x="age", kde=True, color=palette["secondary"], bins=20, edgecolor="#FFFFFF"
    )
    plt.title("Applicant Age Distribution Profile", pad=15)
    plt.xlabel("Age (Years)")
    plt.ylabel("Applicant Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "age_distribution.png"), dpi=150)
    plt.close()
    
    # 4. Loan Amount Distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=df, x="credit_amount", kde=True, color=palette["primary"], bins=25, edgecolor="#FFFFFF"
    )
    plt.title("Loan / Credit Amount Distribution (Exposure Profile)", pad=15)
    plt.xlabel("Credit Amount (Deutsche Mark - DM)")
    plt.ylabel("Applicant Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loan_amount_distribution.png"), dpi=150)
    plt.close()
    
    # 5. Default vs Non-default comparison (Checking Status impact on Defaults)
    plt.figure(figsize=(8, 5))
    # Calculate default rate per checking account status category
    checking_default = df.groupby("checking_status")["class"].mean().reset_index()
    checking_default["Default Rate (%)"] = checking_default["class"] * 100
    checking_default = checking_default.sort_values(by="Default Rate (%)", ascending=False)
    
    sns.barplot(
        data=checking_default, x="Default Rate (%)", y="checking_status",
        color=palette["accent"], edgecolor="#333333", linewidth=1
    )
    # Add percentage text
    for i, row in checking_default.iterrows():
        plt.text(
            row["Default Rate (%)"] + 1, i,
            f"{row['Default Rate (%)']:.1f}%",
            va="center", ha="left", fontsize=10, weight="bold"
        )
        
    plt.title("Default Rate by Checking Account Status", pad=15)
    plt.xlabel("Default Rate (%) - Higher means more risk")
    plt.ylabel("Checking Account Status")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "default_vs_non_default.png"), dpi=150)
    plt.close()
    
    logging.info("EDA visualizations successfully written to outputs/")
