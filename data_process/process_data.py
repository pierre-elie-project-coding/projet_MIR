from data_process.read_and_plot import read_data_from_text
import pandas as pd
import torch
import numpy as np
from collections import Counter

MAPPING_DICT = {
    -2:0,
    -1:1,
    0:2,
    1:3,
    2:4,
    3:5
}

def mapping_slope_to_index(seq:list[int]):
    return [MAPPING_DICT[element] for element in seq]


def fetch_data_for_training(stop:int|None=None,see_stats:bool=False):
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
        stop=stop,
    )

    # utils
    text2float = lambda x: [np.float32(n) for n in x]
    text2int = lambda x: [np.int32(n) for n in x]

    # read_data processing
    df_data = df["read_data"].apply(text2float)
    inputs_list = df_data.to_list()
    inputs_list = [torch.tensor(input) for input in inputs_list]

    # read_sol processing
    df_sol = df["read_sol"].apply(text2float)
    df_sol = df_sol.apply(text2int)
    df_sol = df_sol.apply(mapping_slope_to_index)
    # To see class repartition
    if see_stats:
        stats = {0:0,1:0,2:0,3:0,4:0,5:0}
        for row in df_sol:
            stat_row = dict(Counter(row))
            for key in stats.keys():
                stats[key]+=stat_row[key] if key in stat_row.keys() else 0 
        print(f" Stats : {stats}")
    targets_list = df_sol.to_list()
    targets_list = [torch.tensor(target) for target in targets_list]
    return inputs_list,targets_list

if __name__ == "__main__":
    fetch_data_for_training()
