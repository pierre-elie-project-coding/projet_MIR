import numpy as np
import xgboost as xgb
from scipy.signal import medfilt

class XGBoostStatePredictor:
    def __init__(self, **kwargs):
        """
        Predictor using XGBoost to classify the states from sliding window signals.
        """
        n_estimators = kwargs.get('n_estimators', 150)
        max_depth = kwargs.get('max_depth', 5)
        learning_rate = kwargs.get('learning_rate', 0.1)
        subsample = kwargs.get('subsample', 0.8)
        colsample_bytree = kwargs.get('colsample_bytree', 0.8)
        
        self.model = xgb.XGBClassifier(
            objective='multi:softmax',
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=-1
        )
        self.state_classes = None
        self.state_to_idx = {}
        self.idx_to_state = {}

    def fit(self, X, y):
        """
        Fit the XGBoost model on sliding windows and map states to indices.
        """
        self.state_classes = np.unique(y)
        self.state_to_idx = {state: idx for idx, state in enumerate(self.state_classes)}
        self.idx_to_state = {idx: state for idx, state in enumerate(self.state_classes)}
        
        # XGBoost requires labels to be 0 to num_class - 1
        y_mapped = np.array([self.state_to_idx[val] for val in y])
        self.model.fit(X, y_mapped)

    def predict(self, X):
        """
        Predict states and map them back to original state values.
        """
        preds_idx = self.model.predict(X)
        preds = np.array([self.idx_to_state[idx] for idx in preds_idx])
        return preds

    def predict_proba(self, X):
        """
        Return probabilities for each class.
        """
        return self.model.predict_proba(X)

    def smooth_piecewise_constant(self, predictions, window_size=30, threshold=0.5):
        """
        Post-processing: smooth predictions into piecewise constant plateaus.
        """
        # 1. Median filter (increased kernel size for robustness against noisy predictions)
        smoothed = medfilt(predictions, kernel_size=11)
        
        # 2. Change point detection & piecewise constant filling
        result = np.copy(smoothed)
        n = len(result)
        
        start_idx = 0
        while start_idx < n:
            end_idx = min(start_idx + window_size, n)
            # Find next significant change
            for i in range(start_idx + 1, n):
                window_before = np.mean(smoothed[max(0, i-window_size):i])
                window_after = np.mean(smoothed[i:min(n, i+window_size)])
                if abs(window_after - window_before) > threshold:
                    end_idx = i
                    break
            else:
                end_idx = n
            
            # Fill with median
            segment_val = np.median(smoothed[start_idx:end_idx])
            result[start_idx:end_idx] = segment_val
            start_idx = end_idx
            
        return result
