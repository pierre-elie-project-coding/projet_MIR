"""
MLP using a sliding window
Naive approach
"""

import torch.nn as nn


class MlpSlidingWindow(nn.Module):
    def __init__(self, window_size: int) -> None:
        super().__init__()
        self.first_linear_relu_stack = nn.Sequential(
            nn.Linear(window_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        self.second_linear_relu_stack = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
        )

    def forward(self, x):
        logits = self.first_linear_relu_stack(x)
        logits = self.dropout(logits)
        logits = self.second_linear_relu_stack(logits)
        return logits
