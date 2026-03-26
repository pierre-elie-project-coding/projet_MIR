import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_data_from_text(
    path_read_data: str, path_read_par: str | None = None, path_read_sol: str| None = None, stop: int | None = None, not_below:int | None = None
) -> pd.DataFrame:
    df = pd.DataFrame(columns=["read_id", "read_data", "read_par", "read_sol"])

    # reading read_id and read_data
    with open(path_read_data, "r", encoding="utf-8") as f:
        file = f.read().split("\n")
        if file[-1] == [""]:
            file.pop()
        for i in range(int(len(file) / 2)):
            if stop and (2 * i >= stop):
                break
            if not_below and len(file[2*i+1].split()) < not_below:
                continue
            df.loc[i] = [file[2 * i], file[2 * i + 1].split(), pd.NA, pd.NA]
    df = df.set_index("read_id")

    # reading read_par
    if path_read_par:
        with open(path_read_par, "r", encoding="utf-8") as f:
            file = f.read().split("\n")
            if file[-1] == [""]:
                file.pop()
            for i in range(len(file)):
                if i == stop:
                    break
                result = file[i].split("{", 1)
                id = result[0].strip()
                dict_data = "{" + result[1].strip()
                if id in df.index:
                    df.at[id, "read_par"] = ast.literal_eval(dict_data)

    # reading read_sol
    if path_read_sol:
        with open(path_read_sol, "r", encoding="utf-8") as f:
            file = f.read().split("\n")
            if file[-1] == [""]:
                file.pop()
            for i in range(int(len(file) / 2)):
                if stop and 2 * i >= stop:
                    break
                id = file[2 * i]
                val = file[2 * i + 1].split()
                if id in df.index:
                    df.at[id, "read_sol"] = val # type: ignore

    return df


def build_affine_signal(signal: list[int]) -> list[int]:
    """
    1=monte 2= descend  ( -1=monte -2= descend mais dans l'autre sens) 3=plateau haut 0= plateau bas
    """
    step = 1  # TODO :change this value and link it to reality
    affine_signal = [] 
    current_val = 0

    for v in signal:
        if v in [1, 2]:           # Pente positive du profil temporel
            current_val += step
        elif v in [-1, -2]:       # Pente négative du profil temporel
            current_val -= step
        # Si v == 0 ou v == 3, current_val ne change pas
        
        affine_signal.append(current_val)

    return affine_signal


def plot_one_data(df: pd.DataFrame, element: str, with_time_profile: bool = False):
    row = df.loc[element]
    abs = np.linspace(start=0, stop=len(row["read_data"]), num=len(row["read_data"]))
    z_value = list(map(float, row["read_data"]))
    print(f"READ_PAR : \n {row['read_par']}")
    plt.figure()
    plt.plot(abs, z_value, label="z profile")
    plt.title(f"Element : {element}")
    plt.legend()
    if with_time_profile:
        plt.figure()
        time_value = list(map(int, row["read_sol"]))
        affine_signal = build_affine_signal(signal=time_value)
        plt.plot(affine_signal, linestyle="--", color="red", label="time profile")
        plt.title(f"Element : {element}")
        plt.legend()

    plt.show()
