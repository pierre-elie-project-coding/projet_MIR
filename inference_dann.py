import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# --- Project specific imports ---
from utils.parse_config import get_config
from data_process.read_and_plot import build_affine_signal, read_data_from_text
from data_process.process_data import fetch_data_for_training

# Mapping from classes (0-5) back to slopes (-2, -1, 0, 1, 2, 3)
INDEX_TO_SLOPE = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2, 5: 3}

class DANN(nn.Module):
    def __init__(self, window_size):
        super(DANN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(window_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x, alpha=None):
        feature = self.feature(x)
        if alpha is not None:
            return self.class_classifier(feature), self.domain_classifier(feature)
        else:
            return self.class_classifier(feature)

def load_dann_model(window_size, weights_path="weights/dann/dann_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DANN(window_size=window_size).to(device)
    
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded DANN weights from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}. Inference will use random weights.")
    
    model.eval()
    return model, device

def load_real_data(real_data_path, seed=42, number_real_data=5):
    df = pd.read_pickle(real_data_path)
    text2float = lambda x: [np.float32(n) for n in x]
    df_data = df["noisy_read"].apply(text2float)
    # Shuffle with the same seed as show_results.py
    df_data = shuffle(df_data, random_state=seed)
    inputs_list = df_data.to_list()
    inputs_list = [torch.tensor(input) for input in inputs_list]
    return inputs_list[:number_real_data]

def predict_single_signal(model, device, signal_1d, window_size=51, batch_size=256):
    model.eval()
    padding_size = int((window_size - 1) / 2)
    start = signal_1d[:padding_size].flip(dims=[0])
    end = signal_1d[-padding_size:].flip(dims=[0])
    padded_signal = torch.cat((start, signal_1d, end), dim=0).to(torch.float32)

    windows = padded_signal.unfold(0, window_size, step=1)
    num_windows = windows.shape[0]
    all_predictions = []

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch_windows = windows[i : i + batch_size].to(device)
            # Robust normalization as used in DANN training
            b_mean = batch_windows.mean(dim=1, keepdim=True)
            b_std = batch_windows.std(dim=1, keepdim=True) + 1e-8
            batch_windows = (batch_windows - b_mean) / b_std
            
            logits = model(batch_windows)
            preds = torch.argmax(logits, dim=1)
            all_predictions.append(preds.cpu())

    final_predictions = torch.cat(all_predictions, dim=0)
    return final_predictions.numpy()

def inference_dann(seed=25): # Using seed 25 by default as in show_results.py
    config = get_config()
    window_size = config["model"]["mlp"]["sliding_window_size"]
    real_data_path = "data/BT10_100uM_RefBT1multi_analysis_E.df"
    
    model, device = load_dann_model(window_size)
    
    # 1. Load Real Data (Target)
    inputs_real = load_real_data(real_data_path, seed=seed, number_real_data=5)
    
    # 2. Load Simulated Data (Source) - Use same logic as show_results
    inputs_simulated, targets_simulated, _ = fetch_data_for_training(stop=5)
    
    os.makedirs("graphics", exist_ok=True)

    # --- SIMULATED DATA INFERENCE ---
    print(f"DANN inference on simulated data (Seed: {seed})")
    for index, sig in enumerate(inputs_simulated):
        pred = predict_single_signal(model, device, sig, window_size=window_size)
        size_el = pred.shape[0]
        well_formated_pred = [INDEX_TO_SLOPE[int(i)] for i in pred]
        well_formated_target = [INDEX_TO_SLOPE[int(i)] for i in targets_simulated[index]]
        x = np.arange(0, size_el, 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))
        ax1.plot(x, well_formated_target, color="green")
        ax1.plot(x, well_formated_pred, linestyle='--', color="orange")
        ax1.set_title("Source of Truth (Simulated)")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Slope")

        ax2.plot(x, well_formated_pred, color="orange")
        ax2.set_title("DANN Prediction (Simulated)")
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Slope")

        plt.tight_layout()
        plt.savefig(f"graphics/dann-simulated-{index}.png")
        print(f"Saved simulated result to graphics/dann-simulated-{index}.png")
        plt.close()

    # --- REAL DATA INFERENCE ---
    print(f"DANN inference on real data (Seed: {seed})")
    for index, sig in enumerate(inputs_real):
        pred = predict_single_signal(model, device, sig, window_size=window_size)
        size_el = pred.shape[0]
        x = np.arange(0, size_el, 1)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))
        ax1.plot(x, sig.numpy(), color="green")
        ax1.set_title("Original Real Signal")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Amplitude")

        well_formated_pred = [INDEX_TO_SLOPE[int(i)] for i in pred]
        affine_pred = build_affine_signal(well_formated_pred)
        ax2.plot(x, affine_pred, color="orange") 
        ax2.set_title("DANN Prediction (Real)")
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Reconstructed Slope")

        plt.tight_layout()
        plt.savefig(f"graphics/dann-realdata-{index}.png")
        print(f"Saved real result to graphics/dann-realdata-{index}.png")
        plt.close()

if __name__ == "__main__":
    # Use seed 25 to match show_results.py behavior
    inference_dann(seed=25)
