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
    print(f"READ_PAR : \n {row["read_par"]}")
    plt.figure()
    plt.plot(abs,value,label=element)
    plt.legend()
    plt.show()
    

def main():
    data =read_data_from_text(path_read_data="data/learning_test.fa",path_read_par="data/learning_test_parameters.txt",stop=16)
    print(f"See DF : \n {data.head()}")
    e1 = "c82b24c4-a3a7-4000-b2a6-95bd3815d150"
    e2 = "0659dd4c-cf20-4afd-8674-eb9e6769909d"
    plot_one_data(df=data,element=e2)
    

if __name__ == "__main__":
    main()