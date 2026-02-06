import pandas as pd
from collections import Counter


def get_targets_repartition(df_sol: pd.Series):
    stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for row in df_sol:
        stat_row = dict(Counter(row))
        for key in stats.keys():
            stats[key] += stat_row[key] if key in stat_row.keys() else 0
