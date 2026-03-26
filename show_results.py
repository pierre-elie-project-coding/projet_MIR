from typing import Any

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import gc
from data_process.datasets import LazySlidingWindowDataset, UnetSlidingWindowDataset
from data_process.process_data import fetch_data_for_training
from models.unet import Unet
from models.mlp import MlpSlidingWindow
from utils.parse_config import get_config
from data_process.read_and_plot import build_affine_signal, read_data_from_text

# Mapping from classes (0-5) back to slopes (-2, -1, 0, 1, 2, 3)
INDEX_TO_SLOPE = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2, 5: 3}

def load_model(config,model_name:str|None=None):
    model_name = model_name if model_name else config.get("train", "unet")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "unet":
        sliding_window_size = config["model"]["unet"]["sliding_window_size"]
        padding = config["model"]["unet"]["padding"]
        channels = [64, 128, 256, 512, 1024]
        model = Unet(kernel_size=3, pool_kernel_size=2, channels=channels, kernel_size_upconv=2, padding=padding).to(device)
        weights_path = "weights/unet/unet.pth"
    elif model_name == "mlp":
        sliding_window_size = config["model"]["mlp"]["sliding_window_size"]
        model = MlpSlidingWindow(window_size=sliding_window_size).to(device)
        weights_path = "weights/mlp_sw/mlp_sliding_window_model.pth"
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"Warning: Weights not found at {weights_path}.")
    
    model.eval()
    return model, device, model_name, sliding_window_size

def load_data(real_data_path:str,number_real_data:int=5):
    df = pd.read_pickle(real_data_path)
    text2float = lambda x: [np.float32(n) for n in x]
    df_data = df["noisy_read"].apply(text2float)
    inputs_list = df_data.to_list()
    inputs_list = [torch.tensor(input) for input in inputs_list]
    return inputs_list[:number_real_data]

def predict_single_signal(model, signal_1d, window_size=512, batch_size=256):
    """
    Évalue un signal complet avec une fenêtre glissante sans utiliser de Dataset.
    """
    model.eval()
    # device = next(model.parameters()).device
    # signal_1d = signal_1d.to(device)
    
    padding_size = int((window_size - 1) / 2) + 2
    start = signal_1d[:padding_size].flip(dims=[0])
    end = signal_1d[-padding_size:].flip(dims=[0])
    padded_signal = torch.cat((start, signal_1d, end), dim=0).to(torch.float32)

    windows = padded_signal.unfold(0, window_size, step=1)
    
    windows = windows.unsqueeze(1) #dimension "Channel" (1) pour le U-Net -> (N, 1, window_size)
    
    num_windows = windows.shape[0]
    all_predictions = []
    center_idx = window_size // 2  # L'index du point central

    # 3. Prédiction par mini-batchs (pour ne pas exploser la RAM de la carte graphique)
    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch_windows = windows[i : i + batch_size]
            
            # Sortie du U-net : (Batch, 6 classes, window_size)
            logits = model(batch_windows)
            
            # 4. On extrait uniquement la prédiction pour le point central !
            # logits_center shape : (Batch, 6 classes)
            logits_center = logits[:, :, center_idx]
            
            # On prend la classe max (0 à 5)
            preds = torch.argmax(logits_center, dim=1)
            all_predictions.append(preds.cpu())

    # 5. On recolle tous les batchs pour avoir la prédiction finale de la taille du signal
    final_predictions = torch.cat(all_predictions, dim=0)
    
    # Ajustement si le padding a généré quelques points en trop par rapport au signal brut
    final_predictions = final_predictions[:len(signal_1d)]
    
    return final_predictions.numpy()

def show_results(real_data_path:str,number_training_data:int=5,number_real_data:int=5):
    config = get_config()
    (inputs_simulated, targets_simulated, weights) = fetch_data_for_training(stop=number_training_data)
    inputs_real = load_data(real_data_path=real_data_path,number_real_data=number_real_data)
    model, device, model_name, sliding_window_size = load_model(config=config,model_name=None)


    ## Simulated 
    print(f"Model inference on simulated data")
    preds_simulated = []
    for sig in inputs_simulated:
        preds_simulated.append(predict_single_signal(model=model,signal_1d=sig,window_size=sliding_window_size,batch_size=64))

    ### Plot Simulated
    for index,pred in enumerate(preds_simulated):
        print(pred.shape)
        size_el = pred.shape[0]
        x = np.arange(0, size_el, 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))

        ax1.plot(x, targets_simulated[index], color="green")
        ax1.plot(x,pred,linestyle='--',color="orange")
        ax1.set_title("Source of truth")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Temps")

        # 4. Remplissage du deuxième graphique (à droite)
        ax2.plot(x, pred, color="orange") # J'utilise un diagramme en barres ici pour l'exemple
        ax2.set_title(f"{model_name} Prediction")
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Temps")

        # Ajuste automatiquement les espacements pour éviter que les textes ne se chevauchent
        plt.tight_layout()

        # 5. Affichage
        plt.show()

    ## Real Data
    print(f"Model inference on real data")
    preds_real = []
    for sig in inputs_real:
        preds_real.append(predict_single_signal(model=model,signal_1d=sig,window_size=sliding_window_size,batch_size=64))
    for index,pred in enumerate(preds_real):
        print(pred.shape)
        size_el = pred.shape[0]
        x = np.arange(0, size_el, 1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8))

        ax1.plot(x, inputs_real[index], color="green")
        ax1.set_title("Source of truth")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Temps")

        # 4. Remplissage du deuxième graphique
        # pred = build_affine_signal(pred)
        ax2.plot(x, pred, color="orange") 
        ax2.set_title(f"{model_name} Prediction")
        ax2.set_xlabel("Position")
        ax2.set_ylabel("Temps")

        # Ajuste automatiquement les espacements pour éviter que les textes ne se chevauchent
        plt.tight_layout()

        # 5. Affichage
        plt.show()


    ### Real Data
    # preds_real = model(inputs_real)




if __name__=="__main__":
    show_results(real_data_path="data/BT10_100uM_RefBT1multi_analysis_E.df",number_real_data=5,number_training_data=5)




