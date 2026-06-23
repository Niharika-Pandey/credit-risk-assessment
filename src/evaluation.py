import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
from src.utils import get_palette

def compute_metrics(y_true, y_pred, y_prob):
    """Computes basic classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def evaluate_models(trained_models, X_test, y_test):
    """Evaluates all trained models on the test set and returns a comparison DataFrame."""
    logging.info("Evaluating all models on the test set...")
    metrics_summary = {}
    
    for name, model in trained_models.items():
        # Predict class labels
        y_pred = model.predict(X_test)
        
        # Predict probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            # Normalize to [0,1] for AUC
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-5)
            
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics_summary[name] = metrics
        
        logging.info(
            f"{name} Test Metrics -> "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Recall (Sensitivity): {metrics['recall']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )
        
    df_compare = pd.DataFrame(metrics_summary).T
    df_compare.index.name = "Model"
    
    # Save as JSON and CSV
    df_compare.to_csv("outputs/model_comparison.csv")
    with open("outputs/model_comparison.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    logging.info("Model evaluation metrics saved to outputs/")
    return df_compare

def plot_roc_curves(trained_models, X_test, y_test):
    """Plots and saves ROC curves for all models in a single chart."""
    palette = get_palette()
    plt.figure(figsize=(8, 6))
    
    colors = [palette["primary"], palette["secondary"], "#F39C12", palette["accent"]]
    
    for (name, model), color in zip(trained_models.items(), colors):
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min() + 1e-5)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2)
        
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("ROC-AUC Curve Comparison (Test Set)")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    plot_path = "outputs/roc_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"ROC curve comparison plot saved to {plot_path}")

def plot_confusion_matrices(trained_models, X_test, y_test):
    """Plots and saves confusion matrices for all models side-by-side or individually."""
    palette = get_palette()
    
    # Plot for the best performing model (often Random Forest or XGBoost)
    # Let's save a single clean plot for the best model to make it presentation-ready,
    # or save them individually. Let's do it individually for each model.
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6, 5))
        # Custom color map using our success (green) and accent (red) hues, or simple blues
        cmap = sns.light_palette(palette["secondary"], as_cmap=True)
        
        # Labels for cells
        labels = [
            [f"True Creditworthy\n{cm[0,0]}", f"False Default\n{cm[0,1]}"],
            [f"False Creditworthy\n{cm[1,0]}", f"True Default\n{cm[1,1]}"]
        ]
        
        sns.heatmap(
            cm, annot=labels, fmt="", cmap=cmap, cbar=False,
            xticklabels=["Creditworthy (Good)", "Default (Bad)"],
            yticklabels=["Creditworthy (Good)", "Default (Bad)"],
            annot_kws={"size": 11, "weight": "bold"}
        )
        
        plt.ylabel("Actual Borrower Class")
        plt.xlabel("Predicted Borrower Class")
        plt.title(f"Confusion Matrix: {name}")
        plt.tight_layout()
        
        plot_path = f"outputs/confusion_matrix_{name}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
    logging.info("Confusion matrix plots saved to outputs/")

def plot_feature_importances(model, feature_names, model_name, top_n=15):
    """Plots and saves the top feature importances or coefficients of a trained model."""
    palette = get_palette()
    
    # Get importances or coefficients
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Top {top_n} Feature Importances: {model_name}"
    elif hasattr(model, "coef_"):
        # For Logistic Regression, use absolute coefficients
        importances = np.abs(model.coef_[0])
        title = f"Top {top_n} Feature Impact (Absolute Coefficients): {model_name}"
    else:
        logging.warning(f"Model {model_name} does not support feature importances/coefficients.")
        return
        
    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(top_n)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(
        x="Importance", y="Feature", data=df_importance,
        color=palette["secondary"]
    )
    plt.title(title)
    plt.xlabel("Relative Importance Score")
    plt.ylabel("")
    plt.tight_layout()
    
    plot_path = f"outputs/feature_importance_{model_name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logging.info(f"Feature importance plot for {model_name} saved to {plot_path}")
