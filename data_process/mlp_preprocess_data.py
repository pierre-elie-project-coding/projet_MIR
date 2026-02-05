"""
Functions to preprocess data to train a sliding window mlp
"""
import torch
from data_process.process_data import fetch_data_for_training
from data_process.datasets import LazySlidingWindowDataset
from torch.utils.data import TensorDataset, DataLoader, random_split

def preprocess_data_for_mlp(input_tensor:list[torch.Tensor],
                            labels:list[torch.Tensor],
                            sliding_window_size:int=51,
                            batch_size:int=32,
                            shuffle:bool=False,
                            with_split:float=False):

    dataset = LazySlidingWindowDataset(input_tensor=input_tensor,targets=labels,sliding_window_size=sliding_window_size)
    
    if with_split:
        generator = torch.Generator().manual_seed(42)
        train_dataset,test_dataset = random_split(dataset=dataset,lengths=[with_split,1-with_split],generator=generator) # type: ignore
        train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=shuffle)
        test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=shuffle)
        return train_dataloader,test_dataloader
    
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=shuffle) # type: ignore
    return dataloader

if __name__=="__main__":
    inputs,labels = fetch_data_for_training()
    preprocess_data_for_mlp(input_tensor=inputs,labels=labels)
