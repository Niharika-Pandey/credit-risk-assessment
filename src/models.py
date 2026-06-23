import os
import pickle
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Model configuration dictionaries for GridSearchCV
MODEL_GRIDS = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["liblinear", "lbfgs"],
            "class_weight": ["balanced", None]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [3, 5, 7, 10],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 8, 12],
            "min_samples_split": [2, 5],
            "class_weight": ["balanced", "balanced_subsample", None]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "params": {
            "n_estimators": [100, 150],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "scale_pos_weight": [1, 2.3] # 2.3 approximates the class imbalance ratio (70% Good / 30% Bad)
        }
    }
}

def train_and_tune_models(X_train, y_train, scoring_metric="roc_auc", cv_splits=5):
    """Trains multiple classifiers and performs hyperparameter tuning using GridSearchCV."""
    logging.info(f"Starting model training and tuning using metric: {scoring_metric}")
    
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    trained_models = {}
    tuning_results = {}
    
    for model_name, config in MODEL_GRIDS.items():
        logging.info(f"Tuning {model_name}...")
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring=scoring_metric,
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        logging.info(f"{model_name} tuning complete. Best CV {scoring_metric}: {best_score:.4f}")
        logging.info(f"Best parameters: {best_params}")
        
        trained_models[model_name] = best_model
        tuning_results[model_name] = {
            "best_params": best_params,
            "best_cv_score": best_score
        }
        
        # Save each model immediately to pickle
        model_path = f"models/{model_name}_best.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        logging.info(f"Saved best {model_name} model to {model_path}")
        
    # Save tuning summary
    summary_path = "models/tuning_summary.pkl"
    with open(summary_path, "wb") as f:
        pickle.dump(tuning_results, f)
        
    return trained_models, tuning_results
