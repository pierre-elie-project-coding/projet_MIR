# Projet MIR 

## Description
This project focuses on analyzing 1D signals (representing the rate of BrdU incorporation) to predict their temporal profile. The problem is formulated as a 6-class classification task, where the goal is to predict the local slopes (`-2, -1, 0, 1, 2, 3`) of the signal at any given position.

To tackle this problem, the repository implements multiple Deep Learning architectures, including a naive sliding-window approach, a fully convolutional 1D U-Net, and a Domain Adversarial Neural Network (DANN) to bridge the gap between simulated data and real-world noisy data.

## Models Implemented
- **MLP (Multilayer Perceptron):** A naive baseline using a sliding window to predict the central slope.
- **1D U-Net:** A U-Net architecture adapted for 1D signals, consisting of a contractive path (downsampling) and an expansive path (upsampling) with skip connections.
- **DANN (Domain Adversarial Neural Network):** An architecture implementing Unsupervised Domain Adaptation. It uses a gradient reversal layer to align the feature distributions between the simulated data (source domain) and the real data (target domain).

## Project Structure
```text
.
├── config.toml             # Global configuration file (hyperparameters, model selection)
├── main.py                 # Script to load and plot test data
├── train_models.py         # Main training script for MLP and U-Net
├── train_dann.py           # Training script for the DANN architecture
├── show_results.py         # Inference and plotting script for MLP/U-Net
├── inference_dann.py       # Inference and plotting script for DANN
├── data_process/           # Data loading, preprocessing, and PyTorch Datasets
├── models/                 # Neural network definitions (mlp.py, unet.py, etc.)
├── utils/                  # Helper functions (config parsing, saving metrics)
├── weights/                # Directory where trained model weights are saved
└── graphics/               # Directory where inference plots are exported
```

## Prerequisites
- **Python:** `>= 3.13`
- **Package Manager:** `uv` (recommended) or `pip`.

## Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd projet_MIR
   ```

2. Install dependencies using `uv` (or pip):
   ```bash
   uv sync
   # Alternatively: pip install -e .
   ```

## Configuration
The core parameters are centralized in `config.toml`. 
Before running the scripts, you can edit this file to change the model (`train = "unet"` or `train = "mlp"`), adjust hyperparameters (batch size, epochs, learning rate, window size), and specify the metrics output file.

## Usage

### 1. Data Visualization
To run a quick test that reads the data from text files and plots a specific element:
```bash
python main.py
```

### 2. Training MLP or U-Net
1. Open `config.toml` and set `train = "mlp"` or `train = "unet"`.
2. Configure the training parameters in the respective `[training.mlp]` or `[training.unet]` sections.
3. Run the training script:
```bash
python train_models.py
```
*Trained weights will be saved in `weights/mlp_sw/` or `weights/unet/`, and metrics will be appended to `results.md`.*

### 3. Training the DANN
To train the Domain Adversarial Neural Network using simulated data as the source and real data as the target:
```bash
python train_dann.py
```
*The model will be saved in `weights/dann/dann_model.pth`.*

### 4. Inference and Plotting
To evaluate a trained model and generate visual comparisons between predictions and the ground truth (saved as PNGs in the `graphics/` folder):

- **For MLP or U-Net:**
  ```bash
  python show_results.py
  ```
- **For DANN:**
  ```bash
  python inference_dann.py
  ```

## Acknowledgments
Uses `PyTorch` for deep learning, `Pandas` and `NumPy` for data manipulation, and `Matplotlib` for generating visual insights.
