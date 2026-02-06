import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt


def read_data_from_text(path_read_data:str,path_read_par:str,stop:int|None = None)->pd.DataFrame:
    df = pd.DataFrame(columns=["read_id","read_data","read_par"])
    
    # reading read_id and read_data
    with open(path_read_data,"r",encoding="utf-8") as f:
        file = f.read().split("\n")
        for i in range(int(len(file)/2)):
            if 2*i == stop:
                break
            df.loc[i] = [file[2*i],file[2*i+1].split(),pd.NA]
    df = df.set_index("read_id")
    
    # reading read_par
    with open(path_read_par,"r",encoding="utf-8") as f:
        file = f.read().split("\n")
        for i in range(len(file)):
            if i == stop:
                break
            result = file[i].split("{",1)
            id = result[0].strip()
            dict = "{"+result[1].strip()
            if id in df.index:
                df.at[id,"read_par"] = ast.literal_eval(dict)

    return df

def plot_one_data(df:pd.DataFrame,element:str):
    row = df.loc[element]
    abs = np.linspace(start=0,stop=len(row["read_data"]),num=len(row["read_data"]))
    value = list(map(float,row["read_data"]))
    print(f"READ_PAR : \n {row['read_par']}")
    plt.figure()
    plt.plot(abs,value,label=element)
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
import os

def compare_data(id):
    base_path = "data"
    path_data = os.path.join(base_path, "learning_test.fa")
    path_par = os.path.join(base_path, "learning_test_parameters.txt")
    path_states = os.path.join(base_path, "learning_test_states.fa")

    data = read_data_from_text(path_read_data=path_data, path_read_par=path_par, stop=16)
    row = data.loc[id]
    read_data_array = row['read_data']
    read_par = row['read_par']

    values_par = []

    with open(path_states) as f:
        lines_par = f.readlines()

    for i in range(0, len(lines_par), 2):
        read_id_line = lines_par[i].strip().replace('>', '')
        if str(read_id_line) == str(id):
            values_par = list(map(float, lines_par[i+1].split()))
            break

    plt.figure(figsize=(10, 6))
    plt.plot(read_data_array, label='Read Data')
    plt.plot(values_par, linestyle='--', color='red', label='States')
    plt.title(f"ID: {id}")
    print(f"Paramètres: {read_par}") 
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
