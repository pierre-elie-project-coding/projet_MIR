import os
import torch
import numpy as np
import time
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
import xgboost as xgb
print(f"XGBOOST SOURCE LOCATED AT: {xgb.__file__}")
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_process.process_data import fetch_data_for_training
from models.train_xgboost import preprocess_data_for_xgboost
from utils.parse_config import get_config

def optimize_params(stop=None):
    config = get_config()
    
    xgb_config_model = config.get("model", {}).get("xgboost", {})
    sliding_window_size = xgb_config_model.get("sliding_window_size", 11)
    
    xgb_config_training = config.get("training", {}).get("xgboost", {})
    with_split = xgb_config_training.get("with_split", 0.8)
    seed = config.get("seed", 42)
    
    print("=" * 150)
    print("Fetching data for XGBoost Parameter Optimization...")
    inputs, labels, weights = fetch_data_for_training(stop=stop, return_weights_for_loss=False)
    
    print(f"Preprocessing data into sliding windows of size {sliding_window_size} (with statistical features)...")
    X, y = preprocess_data_for_xgboost(inputs, labels, sliding_window_size)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=with_split, random_state=seed
    )
    
    # State mapping
    state_classes = np.unique(y_train)
    state_to_idx = {state: idx for idx, state in enumerate(state_classes)}
    y_train_mapped = np.array([state_to_idx[val] for val in y_train])
    y_test_mapped = np.array([state_to_idx[val] for val in y_test])
    
    print(f"Setting up RandomizedSearchCV for {len(X_train)} windows...")
    
    param_dist = {
        'n_estimators': randint(50, 300),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5)
    }

    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(state_classes),
        n_jobs=-1,
        eval_metric='mlogloss',
        random_state=seed
    )
    
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=10,  # Modify to a higher value for more thorough search
        scoring='f1_macro',
        cv=3,
        verbose=2,
        n_jobs=1, # XGBoost runs internally in parallel, multiple threads here can cause memory spike
        random_state=seed
    )
    
    start_time = time.time()
    random_search.fit(X_train, y_train_mapped)
    
    print(f"\nOptimization finished in {time.time() - start_time:.2f} seconds.")
    print("Best parameters found: ", random_search.best_params_)
    print(f"Best CV F1-macro score: {random_search.best_score_:.4f}")
    
    # Evaluate best model on test set
    best_model = random_search.best_estimator_
    preds_mapped = best_model.predict(X_test)
    
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(y_test_mapped, preds_mapped)
    f1 = f1_score(y_test_mapped, preds_mapped, average='macro')
    
    print(f"Test Error with best params: \n F1-score: {(100*f1):>0.1f}% Accuracy: {(100*acc):>0.1f}% \n")

if __name__ == "__main__":
    optimize_params(stop=200) # Testing with max dataset subset for the optimizer

