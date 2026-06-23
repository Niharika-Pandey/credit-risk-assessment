import logging
import pandas as pd
from src.utils import setup_logging, create_directories, apply_plot_theme
from src.data_loader import load_and_map_data
from src.eda import generate_eda_visualizations
from src.feature_engineering import engineer_features
from src.preprocessing import preprocess_and_split
from src.models import train_and_tune_models
from src.evaluation import evaluate_models, plot_roc_curves, plot_confusion_matrices, plot_feature_importances

def main():
    # 1. Setup logging, folders, and visual theme
    setup_logging()
    logging.info("Starting end-to-end Credit Risk Assessment Pipeline...")
    create_directories()
    apply_plot_theme()
    
    # 2. Acquire and map dataset
    raw_df = load_and_map_data()
    logging.info(f"Loaded dataset with shape: {raw_df.shape}")
    
    # 3. Perform EDA and save professional charts
    generate_eda_visualizations(raw_df)
    
    # 4. Feature engineering
    engineered_df = engineer_features(raw_df)
    
    # 5. Preprocess and Split Data (removes duplicates, caps outliers, standardizes, encodes)
    # The preprocessor is fitted on Train and applied to Test, then saved as pickle
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(
        engineered_df, target_col="class", test_size=0.2, random_state=42
    )
    
    # 6. Train and Tune Models using GridSearch + Stratified CV
    # Models: Logistic Regression, Decision Tree, Random Forest, XGBoost
    trained_models, tuning_results = train_and_tune_models(
        X_train, y_train, scoring_metric="roc_auc", cv_splits=5
    )
    
    # 7. Evaluate and Compare Models on the Test Set
    df_compare = evaluate_models(trained_models, X_test, y_test)
    print("\n--- Test Set Evaluation Results ---")
    print(df_compare.round(4))
    print("-----------------------------------\n")
    
    # 8. Generate Evaluation Plots (ROC curves, Confusion Matrices)
    plot_roc_curves(trained_models, X_test, y_test)
    plot_confusion_matrices(trained_models, X_test, y_test)
    
    # 9. Find the best model based on ROC-AUC and plot its feature importances
    best_model_name = df_compare["roc_auc"].idxmax()
    logging.info(f"Best model based on Test ROC-AUC: {best_model_name}")
    
    best_model = trained_models[best_model_name]
    plot_feature_importances(best_model, feature_names, best_model_name, top_n=15)
    
    logging.info("Pipeline executed successfully. All models, reports, and plots saved.")

if __name__ == "__main__":
    main()
