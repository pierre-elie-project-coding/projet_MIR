import pandas as pd
from collections import Counter
import torch

def get_targets_repartition(df_sol):
    stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for row in df_sol:
        stat_row = dict(Counter(row))
        for key in stats.keys():
            stats[key] += stat_row[key] if key in stat_row.keys() else 0
    return stats

def build_weights(repartition,power:int=1):
    weights = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    max_w = max(repartition.values())
    for key in repartition.keys():
        weights[key] = (max_w/repartition[key])**(1/power) 
    weight_list = [weights[i] for i in range(len(weights.keys()))]
    return torch.tensor(weight_list).float()
