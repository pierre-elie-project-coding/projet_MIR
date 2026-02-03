from read_and_plot import read_data_from_text
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split


def pad_sequence(seq, max_len):
    """
    To make a sequence padding

    Parameters
    ----------
    seq
    max_len
    """
    return seq + [0.0] * (max_len - len(seq))


def fetch_data_for_training():
    """
    _summary_

    Returns
    -------
    _type_
        _description_
    """
    path = "data/"
    path_read_data = path + "learning_test.fa"
    path_read_par = path + "learning_test_parameters.txt"
    path_read_sol = path + "learning_test_states.fa"

    df = read_data_from_text(
        path_read_data=path_read_data,
        path_read_par=path_read_par,
        path_read_sol=path_read_sol,
        stop=16,
    )
    text2float = lambda x: [np.float32(n) for n in x]

    # read_data processing
    df_data = df["read_data"].apply(text2float)
    max_len = df_data.apply(len).max()
    # TODO : find a different solution as putting 0 ruins the data
    df_data_padded = df_data.apply(
        lambda x: pad_sequence(x, max_len)
    )  # padding so all rows are the same length
    df_data = np.array(df_data_padded.tolist(), dtype=np.float32)

    # read_sol processing
    df_sol = df["read_sol"].apply(text2float)
    max_len = df_sol.apply(len).max()
    # TODO : find a different solution as putting 0 ruins the data
    df_sol_padded = df_sol.apply(
        lambda x: pad_sequence(x, max_len)
    )  # padding so all rows are the same length
    df_sol = np.array(df_sol_padded.tolist(), dtype=np.float32)

    print(f"df data : {df_data.shape}")
    print(f"df sol : {df_sol.shape}")

    inputs_tensor = torch.from_numpy(df_data)
    labels_tensor = torch.from_numpy(df_sol)

    print(f"input tensor : {inputs_tensor.shape}")
    print(f"sol tensor : {labels_tensor.shape}")

    return inputs_tensor,labels_tensor

def tensor_to_dataloader(input_tensor:torch.Tensor,labels:torch.Tensor,batch_size:int=64,shuffle:bool=False,with_split:float=False):
    dataset = TensorDataset(input_tensor,labels)
    
    if with_split:
        generator = torch.Generator().manual_seed(42)
        train_dataset,test_dataset = random_split(dataset=dataset,lengths=[with_split,1-with_split],generator=generator)
        train_dataloader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=shuffle)
        test_dataloader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=shuffle)
        return train_dataloader,test_dataloader
    
    dataloader = DataLoader(dataset=dataset,batch_size=64,shuffle=shuffle)
    return dataloader

if __name__ == "__main__":
    fetch_data_for_training()