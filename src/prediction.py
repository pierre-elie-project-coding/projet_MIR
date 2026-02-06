import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import ast

# Import existing loader
try:
    from load_data import read_data_from_text
except ImportError:
    # Fallback if running directly from src or similar issues
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from load_data import read_data_from_text

class DataProcessor:
    def __init__(self, base_path="data"):
        self.base_path = base_path
        self.path_data = os.path.join(base_path, "learning_test.fa")
        self.path_par = os.path.join(base_path, "learning_test_parameters.txt")
        self.path_states = os.path.join(base_path, "learning_test_states.fa")

    def load_all_data(self, stop=None):
        print("Loading data...")
        # Load main data and parameters
        df = read_data_from_text(self.path_data, self.path_par, stop=stop)
        
        # Load states
        states_dict = {}
        if os.path.exists(self.path_states):
            with open(self.path_states, 'r') as f:
                lines = f.readlines()
                for i in range(0, len(lines), 2):
                    if stop and i > stop * 2: break
                    rid = lines[i].strip().replace('>', '')
                    if rid in df.index:
                        states_dict[rid] = list(map(float, lines[i+1].split()))
        
        # Add states to DF
        df['values_par'] = df.index.map(states_dict)
        
        # Convert read_data from strings to floats
        df['read_data'] = df['read_data'].apply(lambda x: [float(i) for i in x])
        
        return df

    def extract_global_features(self, read_data_list):
        features = []
        for signal in read_data_list:
            sig = np.array(signal)
            f = {
                'mean': np.mean(sig),
                'std': np.std(sig),
                'min': np.min(sig),
                'max': np.max(sig),
                'q25': np.percentile(sig, 25),
                'q50': np.percentile(sig, 50),
                'q75': np.percentile(sig, 75),
                'len': len(sig)
            }
            features.append(f)
        return pd.DataFrame(features)

    def prepare_window_data(self, df, window_size=10):
        X_windows = []
        y_windows = []
        
        for idx, row in df.iterrows():
            signal = row['read_data']
            states = row['values_par']
            
            if not isinstance(states, list) or len(states) != len(signal):
                continue
                
            # Sliding window
            # Pad signal for the beginning? Or just start from window_size
            # Simple approach: predict state[i] using signal[i-window:i+window] or signal[i-window:i]
            # Let's use a centered window if possible, or past context
            # Given the request "trouver read_par et values_par", let's assume causality might not be strict or we use local context.
            # Local context centered: [i-w, i+w]
            
            sig_arr = np.array(signal)
            pad_width = window_size // 2
            sig_padded = np.pad(sig_arr, pad_width, mode='edge')
            
            for i in range(len(states)):
                # Window centered on i
                window = sig_padded[i : i + window_size]
                X_windows.append(window)
                y_windows.append(states[i])
                
        return np.array(X_windows), np.array(y_windows)

class XGBoostPredictor:
    def __init__(self):
        self.param_models = {}
        self.state_model = None
        # State classes will be determined from training data
        self.state_classes = None
        self.state_to_idx = {}
        self.idx_to_state = {}

    def train_param_models(self, X, y_df):
        # y_df contains columns for each parameter
        for col in y_df.columns:
            print(f"Training model for parameter: {col}")
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model.fit(X, y_df[col])
            self.param_models[col] = model

    def train_state_model(self, X, y):
        print("Training state model (multi-class classification)...")
        
        # Discover unique classes in the training data
        unique_states = np.unique(y)
        print(f"Unique classes in training: {unique_states}")
        
        # Create mapping: state value -> 0-based index for XGBoost
        self.state_classes = unique_states
        self.state_to_idx = {int(s): i for i, s in enumerate(unique_states)}
        self.idx_to_state = {i: int(s) for i, s in enumerate(unique_states)}
        
        # Encode labels to 0-based indices
        y_encoded = np.array([self.state_to_idx[int(val)] for val in y])
        
        self.state_model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(self.state_classes),
            n_estimators=150,
            max_depth=5
        )
        self.state_model.fit(X, y_encoded)

    def predict_params(self, X):
        preds = {}
        for col, model in self.param_models.items():
            preds[col] = model.predict(X)
        return pd.DataFrame(preds)

    def predict_states(self, X):
        # Predict class indices and decode to actual state values
        y_pred_idx = self.state_model.predict(X)
        return np.array([self.idx_to_state[int(idx)] for idx in y_pred_idx])
    
    def smooth_piecewise_constant(self, predictions, window_size=20, threshold=0.3):
        """
        Apply piecewise constant smoothing to predictions.
        For classification, this mainly helps remove outlier predictions.
        
        Args:
            predictions: Predictions from the classifier (already discrete)
            window_size: Size of window for detecting changes
            threshold: Threshold for detecting significant changes
        
        Returns:
            Smoothed predictions with constant segments
        """
        smoothed = np.copy(predictions)
        n = len(predictions)
        
        # First pass: median filter for denoising
        from scipy.ndimage import median_filter
        smoothed = median_filter(predictions, size=5)
        
        # Second pass: detect change points and create constant segments
        segments = []
        current_start = 0
        
        for i in range(window_size, n, window_size // 2):
            # Compare variance before and after
            window_before = smoothed[max(0, i-window_size):i]
            window_after = smoothed[i:min(n, i+window_size)]
            
            if len(window_before) > 0 and len(window_after) > 0:
                mean_before = np.mean(window_before)
                mean_after = np.mean(window_after)
                
                # Detect significant change
                if abs(mean_after - mean_before) > threshold:
                    segments.append((current_start, i, np.median(smoothed[current_start:i])))
                    current_start = i
        
        # Add final segment
        if current_start < n:
            segments.append((current_start, n, np.median(smoothed[current_start:n])))
        
        # Apply constant values to segments
        result = np.zeros_like(predictions)
        for start, end, value in segments:
            result[start:end] = value
            
        return result

def main():
    # Setup
    processor = DataProcessor(base_path="data") # Assuming running from project root
    
    # Load Data
    # Use a small stop for testing if needed, or None for full
    df = processor.load_all_data(stop=200) 
    print(f"Loaded {len(df)} samples")
    
    # --- Parameter Prediction ---
    print("\n--- Parameter Prediction ---")
    X_params = processor.extract_global_features(df['read_data'])
    
    # Extract targets from dictionary, handling list-valued parameters
    first_par = df['read_par'].iloc[0]
    if isinstance(first_par, dict):
        # Convert to DataFrame
        y_params_raw = pd.DataFrame(df['read_par'].tolist())
        
        # Handle list-valued columns by taking the mean (or first value if empty)
        y_params = pd.DataFrame()
        for col in y_params_raw.columns:
            # Check if this column contains lists
            sample_val = y_params_raw[col].iloc[0]
            if isinstance(sample_val, list):
                # For variable-length lists, use mean (or 0 if empty)
                # This gives us a single value called 'speed' instead of 'speed_mean', 'speed_count', etc.
                y_params[col] = y_params_raw[col].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
            else:
                # Keep scalar values as-is
                y_params[col] = y_params_raw[col]
    else:
        print("No parameters found or wrong format.")
        return

    # Clean data: remove rows with NaN values
    print(f"Before cleaning: {len(y_params)} samples")
    valid_mask = ~y_params.isna().any(axis=1) & ~X_params.isna().any(axis=1)
    X_params = X_params[valid_mask].reset_index(drop=True)
    y_params = y_params[valid_mask].reset_index(drop=True)
    print(f"After cleaning: {len(y_params)} samples")
    
    if len(y_params) == 0:
        print("No valid samples after cleaning!")
        return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_params, y_params, test_size=0.2, random_state=42)
    
    predictor = XGBoostPredictor()
    predictor.train_param_models(X_train, y_train)
    
    # Evaluate
    preds_params = predictor.predict_params(X_test)
    for col in y_params.columns:
        rmse = np.sqrt(mean_squared_error(y_test[col], preds_params[col]))
        print(f"Parameter '{col}' RMSE: {rmse:.4f}")

    # --- State Prediction ---
    print("\n--- State Prediction ---")
    # This prepares individual windows (many samples per read)
    # Using small subset for speed in dev
    subset_df = df.iloc[:20] 
    window_size = 10
    X_states, y_states = processor.prepare_window_data(subset_df, window_size=window_size)
    
    print(f"State training samples: {len(X_states)}")
    X_st_train, X_st_test, y_st_train, y_st_test = train_test_split(X_states, y_states, test_size=0.2, random_state=42)
    
    predictor.train_state_model(X_st_train, y_st_train)
    
    preds_states = predictor.predict_states(X_st_test)
    accuracy = accuracy_score(y_st_test, preds_states)
    print(f"State prediction Accuracy: {accuracy:.4f}")
    print(f"Predicted class distribution: {np.unique(preds_states, return_counts=True)}")

    # --- Visualization of one example ---
    print("\nVisualizing one test example...")
    test_id = df.index[1]
    row = df.loc[test_id]
    
    # Predict params
    feat = processor.extract_global_features([row['read_data']])
    pred_p = predictor.predict_params(feat)
    print(f"ID: {test_id}")
    print(f"True Params: {row['read_par']}")
    print(f"Pred Params: {pred_p.iloc[0].to_dict()}")
    
    # Predict states (reconstruct full sequence)
    sig_arr = np.array(row['read_data'])
    pad_width = window_size // 2
    sig_padded = np.pad(sig_arr, pad_width, mode='edge')
    windows = []
    for i in range(len(sig_arr)):
        windows.append(sig_padded[i : i + window_size])
    windows = np.array(windows)
    pred_s = predictor.predict_states(windows)
    
    # Apply piecewise constant smoothing
    pred_s_smooth = predictor.smooth_piecewise_constant(pred_s, window_size=30, threshold=0.5)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(row['read_data'], label='Read Data')
    plt.title(f"Signal (ID: {test_id})")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    if isinstance(row['values_par'], list):
        plt.plot(row['values_par'], label='True States', color='green', linewidth=2)
    plt.plot(pred_s, label='Raw Pred', linestyle=':', color='orange', alpha=0.5)
    plt.plot(pred_s_smooth, label='Smoothed Pred', linestyle='--', color='red', linewidth=2)
    plt.title("States Prediction (Piecewise Constant)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('prediction_result2.png')
    print("Saved plot to prediction_result.png")

if __name__ == "__main__":
    main()
