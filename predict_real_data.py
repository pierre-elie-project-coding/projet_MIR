"""
Script to train XGBoost on simulated training data and predict states on real data.
Uses the best hyperparameters from results.md.
"""

import os
import numpy as np
import pandas as pd
import torch
import pickle
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, log_loss

from data_process.process_data import fetch_data_for_training
from models.train_xgboost import preprocess_data_for_xgboost
from models.xgboost_model import XGBoostStatePredictor
from utils.parse_config import get_config
from data_process.read_and_plot import build_affine_signal

# Reverse mapping: index -> original state value
# MAPPING_DICT = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5}
INDEX_TO_STATE = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2, 5: 3}
STATE_LABELS = {
    -2: "descend (inv)",
    -1: "monte (inv)",
    0: "plateau bas",
    1: "monte",
    2: "descend",
    3: "plateau haut",
}


def preprocess_real_signal(signal_list, window_size):
    """
    Preprocess a single real signal using the same sliding window + features
    as the training pipeline.
    
    Parameters
    ----------
    signal_list : list of float
        The raw signal values.
    window_size : int
        Size of the sliding window.
    
    Returns
    -------
    np.ndarray
        Feature matrix (n_samples, n_features).
    """
    sig = torch.tensor(signal_list, dtype=torch.float32)
    
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1
    
    # Mirror padding
    start = sig[:pad_left].flip(dims=[0]) if pad_left > 0 else torch.tensor([])
    end = sig[-pad_right:].flip(dims=[0]) if pad_right > 0 else torch.tensor([])
    padded_sig = torch.cat((start, sig, end), dim=0).numpy()
    
    n = len(signal_list)
    X_list = []
    
    for j in range(n):
        window = padded_sig[j : j + window_size]
        
        # Same engineered features as training
        mean_val = np.mean(window)
        std_val = np.std(window)
        min_val = np.min(window)
        max_val = np.max(window)
        p25 = np.percentile(window, 25)
        p75 = np.percentile(window, 75)
        median_val = np.median(window)
        
        features = np.concatenate([
            window,
            [mean_val, std_val, min_val, max_val, p25, p75, median_val]
        ])
        X_list.append(features)
    
    return np.array(X_list)


def train_model(config):
    """
    Train XGBoost on training data with best hyperparameters.
    Returns the trained model and evaluation metrics.
    """
    # Get config
    xgb_config_model = config.get("model", {}).get("xgboost", {})
    sliding_window_size = xgb_config_model.get("sliding_window_size", 256)
    
    xgb_config_training = config.get("training", {}).get("xgboost", {})
    stop = xgb_config_training.get("stop", 512)
    with_split = xgb_config_training.get("with_split", 0.8)
    n_estimators = xgb_config_training.get("n_estimators", 1000)
    max_depth = xgb_config_training.get("max_depth", 7)
    learning_rate = xgb_config_training.get("learning_rate", 0.05)
    subsample = xgb_config_training.get("subsample", 0.8514)
    colsample_bytree = xgb_config_training.get("colsample_bytree", 0.6923)
    seed = config.get("seed", 42)
    
    print("=" * 80)
    print("Step 1: Training XGBoost on simulated/training data")
    print("=" * 80)
    
    # Fetch training data
    print(f"Fetching training data (stop={stop})...")
    inputs, labels, _ = fetch_data_for_training(stop=stop, return_weights_for_loss=False)
    
    # Preprocess
    print(f"Preprocessing with sliding window size = {sliding_window_size}...")
    X, y = preprocess_data_for_xgboost(inputs, labels, sliding_window_size)
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=with_split, random_state=seed
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    
    # Train
    print(f"\nTraining XGBoost (n_estimators={n_estimators}, max_depth={max_depth}, "
          f"lr={learning_rate}, subsample={subsample}, colsample_bytree={colsample_bytree})...")
    
    start_time = time.time()
    model = XGBoostStatePredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    print(f"Training completed in {elapsed:.2f}s")
    
    # Evaluate on validation set
    preds = model.predict(X_val)
    pred_probs = model.predict_proba(X_val)
    
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds, average='macro')
    loss_val = log_loss(y_val, pred_probs)
    
    print(f"\n--- Validation Results ---")
    print(f"  Accuracy : {100*acc:.1f}%")
    print(f"  F1-score : {100*f1:.1f}%")
    print(f"  Log-loss : {loss_val:.6f}")
    print()
    
    return model, sliding_window_size


def predict_real_data(model, sliding_window_size, config):
    """
    Load real data and predict states using the trained model.
    """
    print("=" * 80)
    print("Step 2: Predicting states on real data")
    print("=" * 80)
    
    # Load real data
    data_path = config.get("data", {}).get("path", "data/")
    real_data_file = os.path.join(data_path, "FZ2_RefBT1multi_analysis_E.df")
    
    print(f"Loading real data from {real_data_file}...")
    df_real = pd.read_pickle(real_data_file)
    print(f"Number of signals: {len(df_real)}")
    print(f"Columns: {list(df_real.columns)}")
    print()
    
    # Predict for each signal
    all_predictions = []
    all_predictions_smoothed = []
    
    for idx in range(len(df_real)):
        row = df_real.iloc[idx]
        signal = [float(v) for v in row["noisy_read"]]
        
        # Preprocess
        X_real = preprocess_real_signal(signal, sliding_window_size)
        
        # Predict
        preds = model.predict(X_real)
        preds_smoothed = model.smooth_piecewise_constant(preds)
        
        all_predictions.append(preds.tolist())
        all_predictions_smoothed.append(preds_smoothed.tolist())
        
        if idx % 100 == 0:
            print(f"  Processed {idx+1}/{len(df_real)} signals...")
    
    print(f"  Done! All {len(df_real)} signals processed.")
    
    # Add predictions to dataframe
    df_real["predicted_states"] = all_predictions
    df_real["predicted_states_smoothed"] = all_predictions_smoothed
    
    return df_real


def visualize_predictions(df_real, n_examples=5):
    """
    Visualize some example predictions on real data.
    """
    print("=" * 80)
    print("Step 3: Visualization")
    print("=" * 80)
    
    n_examples = min(n_examples, len(df_real))
    
    fig, axes = plt.subplots(n_examples, 2, figsize=(18, 4 * n_examples))
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_examples):
        row = df_real.iloc[i]
        signal = [float(v) for v in row["noisy_read"]]
        preds_raw = row["predicted_states"]
        preds_smooth = row["predicted_states_smoothed"]
        
        read_id = row.get("read_id", f"Signal {i}")
        
        # Left: signal + raw predictions
        ax1 = axes[i, 0]
        ax1_twin = ax1.twinx()
        
        ax1.plot(signal, color="blue", alpha=0.6, linewidth=0.5, label="Signal (noisy_read)")
        ax1_twin.plot(preds_raw, color="red", alpha=0.7, linewidth=0.8, label="Predicted states (raw)")
        
        ax1.set_title(f"Signal {i} — Raw predictions", fontsize=10)
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Signal value", color="blue")
        ax1_twin.set_ylabel("State", color="red")
        ax1.legend(loc="upper left", fontsize=7)
        ax1_twin.legend(loc="upper right", fontsize=7)
        
        # Right: signal + smoothed predictions
        ax2 = axes[i, 1]
        ax2_twin = ax2.twinx()
        
        ax2.plot(signal, color="blue", alpha=0.6, linewidth=0.5, label="Signal (noisy_read)")
        ax2_twin.plot(preds_smooth, color="green", alpha=0.8, linewidth=1.0, label="Predicted states (smoothed)")
        
        # Also show affine signal reconstruction
        try:
            affine = build_affine_signal([int(v) for v in preds_smooth])
            ax2_twin.plot(np.linspace(0, len(preds_smooth)-1, len(affine)), 
                         affine, color="orange", alpha=0.6, linewidth=0.8, 
                         linestyle="--", label="Affine reconstruction")
        except Exception:
            pass
        
        ax2.set_title(f"Signal {i} — Smoothed predictions", fontsize=10)
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Signal value", color="blue")
        ax2_twin.set_ylabel("State", color="green")
        ax2.legend(loc="upper left", fontsize=7)
        ax2_twin.legend(loc="upper right", fontsize=7)
    
    plt.tight_layout()
    
    output_path = "predictions_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {output_path}")
    plt.close()


def save_predictions(df_real):
    """
    Save predictions to CSV.
    """
    output_path = "real_data_predictions.csv"
    
    # Create a simplified version for CSV export
    df_export = pd.DataFrame()
    df_export["read_id"] = df_real["read_id"]
    df_export["signal_length"] = df_real["noisy_read"].apply(len)
    df_export["predicted_states"] = df_real["predicted_states"].apply(
        lambda x: " ".join(str(int(v)) for v in x)
    )
    df_export["predicted_states_smoothed"] = df_real["predicted_states_smoothed"].apply(
        lambda x: " ".join(str(int(v)) for v in x)
    )
    
    df_export.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    # Also save the full dataframe with predictions as pickle
    pickle_path = "real_data_with_predictions.df"
    df_real.to_pickle(pickle_path)
    print(f"Full DataFrame with predictions saved to {pickle_path}")


def main():
    config = get_config()
    
    # Step 1: Train model
    model, sliding_window_size = train_model(config)
    
    # Step 2: Predict on real data
    df_real = predict_real_data(model, sliding_window_size, config)
    
    # Step 3: Visualize
    visualize_predictions(df_real, n_examples=5)
    
    # Step 4: Save
    save_predictions(df_real)
    
    print("\n" + "=" * 80)
    print("Done! All predictions completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
