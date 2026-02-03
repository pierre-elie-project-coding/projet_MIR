"""
MLP using a sliding window
Naive approach
"""

from typing import Any
import torch.nn as nn

class MlpSlidingWindow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first_linear_relu_stack = nn.Sequential(
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(0.2)
        self.second_linear_relu_stack = nn.Sequential(
            nn.Linear(64, 32), 
            nn.ReLU(), 
            nn.Linear(32, 6)
        )
        # self.softmax = nn.Softmax()

    def forward(self, x):
        logits = self.first_linear_relu_stack(x)
        logits = self.dropout(logits)
        logits = self.second_linear_relu_stack(logits)
        return logits
