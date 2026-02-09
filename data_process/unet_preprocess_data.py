"""
Functions to preprocess data to train a sliding window unet
"""

import torch
from data_process.process_data import fetch_data_for_training
from data_process.datasets import UnetSlidingWindowDataset
from torch.utils.data import TensorDataset, DataLoader, random_split
from utils.parse_config import get_config


def preprocess_data_for_unet(
    input_tensor: list[torch.Tensor], labels: list[torch.Tensor]
):
    config = get_config()
    batch_size = config["training"]["unet"]["batch_size"]
    shuffle = config["training"]["unet"]["shuffle"]
    seed = config["seed"]
    with_split = config["training"]["unet"]["with_split"]
    sliding_window_size = config["model"]["unet"]["sliding_window_size"]

    dataset = UnetSlidingWindowDataset(
        input_tensor=input_tensor,
        targets=labels,
        sliding_window_size=sliding_window_size,
    )

    if with_split:
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(
            dataset=dataset, lengths=[with_split, 1 - with_split], generator=generator #type: ignore
        )  
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        test_dataloader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=shuffle
        )
        return train_dataloader, test_dataloader

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)  # type: ignore
    return dataloader


if __name__ == "__main__":
    inputs, labels, weights = fetch_data_for_training()
    preprocess_data_for_unet(input_tensor=inputs, labels=labels)
