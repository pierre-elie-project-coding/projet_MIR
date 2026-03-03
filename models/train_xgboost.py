import os
import torch
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss

from data_process.process_data import fetch_data_for_training
from utils.parse_config import get_config
from utils.write_output import save_metrics
from models.xgboost_model import XGBoostStatePredictor

def preprocess_data_for_xgboost(input_tensor, labels, window_size):
    """
    Extract sliding windows from input tensors and compute features.
    """
    padding_size = int((window_size - 1) / 2)
    X_list = []
    y_list = []
    
    for i, (sig, target) in enumerate(zip(input_tensor, labels)):
        # Mirror padding for signal
        start = sig[:padding_size].flip(dims=[0])
        end = sig[-padding_size:].flip(dims=[0])
        padded_sig = torch.cat((start, sig, end), dim=0).numpy()
        
        target_np = target.numpy()
        
        # Create sliding windows
        n = len(target_np)
        for j in range(n):
            window = padded_sig[j : j + window_size]
            
            # Engineered features
            mean_val = np.mean(window)
            std_val = np.std(window)
            min_val = np.min(window)
            max_val = np.max(window)
            p25 = np.percentile(window, 25)
            p75 = np.percentile(window, 75)
            median_val = np.median(window)
            
            # Combine raw window and features
            features = np.concatenate([
                window, 
                [mean_val, std_val, min_val, max_val, p25, p75, median_val]
            ])
            
            X_list.append(features)
            y_list.append(target_np[j])
            
    return np.array(X_list), np.array(y_list)

def train_xgboost(stop=None):
    config = get_config()
    
    # Adapt to colleague's config structure or use defaults
    xgb_config_model = config.get("model", {}).get("xgboost", {})
    sliding_window_size = xgb_config_model.get("sliding_window_size", 11)
    
    xgb_config_training = config.get("training", {}).get("xgboost", {})
    with_split = xgb_config_training.get("with_split", 0.8)
    n_estimators = xgb_config_training.get("n_estimators", 150)
    max_depth = xgb_config_training.get("max_depth", 5)
    learning_rate = xgb_config_training.get("learning_rate", 0.1)
    subsample = xgb_config_training.get("subsample", 0.8)
    colsample_bytree = xgb_config_training.get("colsample_bytree", 0.8)
    seed = config.get("seed", 42)
    
    print("=" * 150)
    print("Fetching data for XGBoost...")
    inputs, labels, weights = fetch_data_for_training(stop=stop, return_weights_for_loss=False)
    
    print(f"Preprocessing data into sliding windows of size {sliding_window_size}...")
    X, y = preprocess_data_for_xgboost(inputs, labels, sliding_window_size)
    
    # Train validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=with_split, random_state=seed
    )
    
    print(f"Training XGBoost Classifier on {len(X_train)} windows...")
    start_time = time.time()
    
    model = XGBoostStatePredictor(
        n_estimators=n_estimators, 
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree
    )
    model.fit(X_train, y_train)
    
    print(f"Training finished in {time.time() - start_time:.2f} seconds.")
    print("Evaluating model...")
    preds = model.predict(X_test)
    
    pred_probs = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    loss = log_loss(y_test, pred_probs)
    
    print(f"Test Error: \n F1-score: {(100*f1):>0.1f}% Accuracy: {(100*acc):>0.1f}%, Avg loss: {loss:>8f} \n")
    
    os.makedirs("weights/xgboost", exist_ok=True)
    try:
        with open("weights/xgboost/xgboost_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("Saved XGBoost Model to weights/xgboost/xgboost_model.pkl")
    except Exception as e:
        print(f"Could not save model using pickle: {e}")
    
    # Return metrics formatted as expected by save_metrics in train_models.py
    # Even if train_models.py hasn't specifically called save_metrics yet for xgboost
    epochs_list = [1]
    loss_list = [float(loss)]
    accuracy_list = [float(acc)]
    f1score_list = [float(f1)]
    
    return epochs_list, loss_list, accuracy_list, f1score_list

if __name__ == "__main__":
    train_xgboost()
