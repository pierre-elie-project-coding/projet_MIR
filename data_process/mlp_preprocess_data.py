"""
Functions to preprocess data to train a sliding window mlp
"""
import torch
from process_data import fetch_data_for_training

def preprocess_data_for_mlp(input_tensor:torch.Tensor,labels:torch.Tensor,sliding_window_size:int=101):
    """
    _summary_

    Parameters
    ----------
    input_tensor : torch.Tensor
        _description_
    labels : torch.Tensor
        _description_
    sliding_window_size : int, optional
        _description_, by default 101

    Returns
    -------
    _type_
        _description_
    """
    
    # defining padding size
    padding_size = int((sliding_window_size-1)/2)

    # intializing new tensor each point from a previous signal now becomes an array 
    # shape : (number of signals,sizeof signal, sliding window size minus one)
    shape = (input_tensor.shape[0],input_tensor.shape[1],sliding_window_size-1) 
    tensor_with_sliding_window = torch.zeros(shape)

    for (index,row) in enumerate(input_tensor):

        signal_size = len(row)

        # Mirror padding
        left_padding = row[:padding_size].flip(0)
        right_padding = row[-padding_size:].flip(0)

        padded_row = torch.cat((left_padding,row,right_padding),0)
        row=padded_row

        for point_index in range(padding_size,signal_size+padding_size):
            tensor_with_sliding_window[index,point_index-padding_size]=row[point_index-padding_size:point_index+padding_size]

    print(f"Input tensor shape : {tensor_with_sliding_window.shape}")
    print(f"labels shape : {labels.shape}")
    
    return  tensor_with_sliding_window,labels

if __name__=="__main__":
    inputs,labels = fetch_data_for_training()
    preprocess_data_for_mlp(input_tensor=inputs,labels=labels)
