import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.utils import shuffle
from torchmetrics.classification import MulticlassF1Score

from data_process.process_data import fetch_data_for_training
from utils.parse_config import get_config
from data_process.read_and_plot import read_data_from_text

class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(nn.Module):
    def __init__(self, window_size):
        super(DANN, self).__init__()
        # Feature extractor
        self.feature = nn.Sequential(
            nn.Linear(window_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Label classifier
        self.class_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )
        
        # Domain classifier
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
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            class_output = self.class_classifier(feature)
            return class_output, domain_output
        else:
            class_output = self.class_classifier(feature)
            return class_output

class SourceDataset(Dataset):
    def __init__(self, input_tensor, targets, window_size):
        self.targets = targets
        self.inputs = []
        self.window_size = window_size
        self.padding_size = int((window_size - 1) / 2)

        for input in input_tensor:
            start = input[: self.padding_size].flip(dims=[0])
            end = input[-self.padding_size :].flip(dims=[0])
            self.inputs.append(torch.cat((start, input, end), dim=0))

        self.indices = []
        for i, sig in enumerate(self.targets):
            for j in range(len(sig)):
                self.indices.append((i, j))
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        (i, j) = self.indices[idx]
        signal = self.inputs[i]
        window = signal[j : j + self.window_size]
        target = self.targets[i][j]
        return window.to(torch.float32), target.to(torch.long)

class TargetDataset(Dataset):
    def __init__(self, input_tensor, window_size):
        self.inputs = []
        self.window_size = window_size
        self.padding_size = int((window_size - 1) / 2)

        for input in input_tensor:
            start = input[: self.padding_size].flip(dims=[0])
            end = input[-self.padding_size :].flip(dims=[0])
            self.inputs.append(torch.cat((start, input, end), dim=0))

        self.indices = []
        for i, sig in enumerate(self.inputs):
            original_len = len(sig) - 2 * self.padding_size
            for j in range(original_len):
                self.indices.append((i, j))
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        (i, j) = self.indices[idx]
        signal = self.inputs[i]
        window = signal[j : j + self.window_size]
        return window.to(torch.float32)

def load_real_data(real_data_path, stop=None):
    if not os.path.exists(real_data_path):
        print(f"Warning: {real_data_path} not found.")
        return []
    df = pd.read_pickle(real_data_path)
    text2float = lambda x: [np.float32(n) for n in x]
    df_data = df["noisy_read"].apply(text2float)
    if stop:
        df_data = df_data.iloc[:stop]
    inputs_list = []
    for input_val in df_data.to_list():
        t = torch.tensor(input_val)
        # Handle NaN/Inf in signal
        t = torch.nan_to_num(t, nan=0.0, posinf=1.0, neginf=-1.0)
        inputs_list.append(t)
    return inputs_list

def train_dann():
    config = get_config()
    seed = config["seed"]
    torch.manual_seed(seed)
    
    window_size = config["model"]["mlp"]["sliding_window_size"]
    batch_size = config["training"]["mlp"]["batch_size"]
    lr = 5e-5 # Lower LR for stability
    epochs = config["training"]["mlp"]["epoch"]
    num_workers = config["data"]["num_workers"]
    with_split = config.get("training", {}).get("mlp", {}).get("with_split", 0.8)
    
    real_data_path = "data/BT10_100uM_RefBT1multi_analysis_E.df"
    
    print("Loading source data...")
    source_inputs_raw, source_labels, weights = fetch_data_for_training(stop=config["training"]["mlp"]["stop"])
    source_inputs = [torch.nan_to_num(t, nan=0.0) for t in source_inputs_raw]
    source_dataset_full = SourceDataset(source_inputs, source_labels, window_size)
    
    if with_split:
        train_size = int(with_split * len(source_dataset_full))
        test_size = len(source_dataset_full) - train_size
        source_dataset, val_dataset = random_split(source_dataset_full, [train_size, test_size])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        source_dataset = source_dataset_full
        val_loader = None
        
    source_loader = DataLoader(source_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    print("Loading target data...")
    target_inputs = load_real_data(real_data_path)
    target_dataset = TargetDataset(target_inputs, window_size)
    target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = DANN(window_size=window_size).to(device)
    
    loss_class = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)
    loss_domain = nn.CrossEntropyLoss()
    
    f1_metric = MulticlassF1Score(num_classes=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    len_dataloader = min(len(source_loader), len(target_loader))
    
    for epoch in range(epochs):
        model.train()
        data_source_iter = iter(source_loader)
        data_target_iter = iter(target_loader)
        
        for i in range(len_dataloader):
            p = float(i + epoch * len_dataloader) / (epochs * len_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            try:
                s_img, s_label = next(data_source_iter)
            except StopIteration: break
            # Per-sample normalization
            s_mean = s_img.mean(dim=1, keepdim=True)
            s_std = s_img.std(dim=1, keepdim=True) + 1e-8
            s_img = (s_img - s_mean) / s_std
            s_img, s_label = s_img.to(device), s_label.to(device)
            domain_label_source = torch.zeros(s_img.shape[0]).long().to(device)
            
            class_output, domain_output = model(s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_label_source)
            
            try:
                t_img = next(data_target_iter)
            except StopIteration: break
            # Per-sample normalization
            t_mean = t_img.mean(dim=1, keepdim=True)
            t_std = t_img.std(dim=1, keepdim=True) + 1e-8
            t_img = (t_img - t_mean) / t_std
            t_img = t_img.to(device)
            domain_label_target = torch.ones(t_img.shape[0]).long().to(device)
            
            _, domain_output = model(t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_label_target)
            
            err = err_s_label + err_s_domain + err_t_domain
            
            if torch.isnan(err):
                print("Error: NaN loss detected. Skipping batch.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            err.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if i % 1000 == 0:
                has_nan = False
                for param in model.parameters():
                    if torch.isnan(param).any():
                        has_nan = True
                        break
                if has_nan:
                    print("CRITICAL: Model weights became NaN. Training is broken.")
                    break
            
            if i % 200 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len_dataloader}], "
                      f"Label Loss: {err_s_label.item():.4f}, Domain Loss: {(err_s_domain + err_t_domain).item():.4f}")
            
            if i > 2000: break

        if val_loader:
            model.eval()
            val_loss, correct, total, f1_val = 0, 0, 0, 0
            with torch.no_grad():
                for v_img, v_label in val_loader:
                    v_img, v_label = v_img.to(device), v_label.to(device)
                    # Per-sample normalization
                    v_mean = v_img.mean(dim=1, keepdim=True)
                    v_std = v_img.std(dim=1, keepdim=True) + 1e-8
                    v_img = (v_img - v_mean) / v_std
                    v_output = model(v_img)
                    val_loss += loss_class(v_output, v_label).item()
                    _, predicted = torch.max(v_output.data, 1)
                    total += v_label.size(0)
                    correct += (predicted == v_label).sum().item()
                    f1_val += f1_metric(v_output, v_label).item()
            
            val_loss /= len(val_loader)
            accuracy = 100 * correct / total
            f1_val /= len(val_loader)
            print(f"--- Validation Epoch {epoch+1} ---")
            print(f"Source Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, F1: {f1_val:.4f}")

    os.makedirs("weights/dann", exist_ok=True)
    torch.save(model.state_dict(), "weights/dann/dann_model.pth")
    print("DANN training complete. Model saved to weights/dann/dann_model.pth")

if __name__ == "__main__":
    train_dann()
