import pandas as pd
import os

df_path = "data/FZ2_RefBT1multi_analysis_E.df"
if os.path.exists(df_path):
    df = pd.read_pickle(df_path)
    print(df.columns)
    print(df.iloc[0])
else:
    print(f"{df_path} not found")
