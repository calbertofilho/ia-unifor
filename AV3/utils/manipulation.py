import os
import platform
import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot
from pathlib import Path

def clearScreen() -> None:
    os.system("cls" if (platform.system() == "Windows") else "clear")  # 'Darwin' <- macOS, 'Linux', 'Windows'

def sign(number: float) -> int:
    return 1 if number >= 0 else -1

def loadData(file_name: str, columns: list, separator: str, ignore_header: bool) -> pd.DataFrame:
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dados"))
    if ignore_header:
        return pd.read_csv(file_name, names=columns, sep=separator, skiprows=[0])
    return pd.read_csv(file_name, names=columns, sep=separator)

def shuffleData(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sample(frac=1).reset_index(drop=True)
    return data

def partitionData(data: pd.DataFrame, perc: float) -> tuple[num.ndarray[float], num.ndarray[float], num.ndarray[float], num.ndarray[float]]:
    X = data.iloc[:int((data.tail(1).index.item()+1)*perc), :len(data.axes[1])-1].values
    y = data.iloc[:int((data.tail(1).index.item()+1)*perc), len(data.axes[1])-1:].values
    X_rest = data.iloc[int((data.tail(1).index.item()+1)*perc):, :len(data.axes[1])-1].values
    y_rest = data.iloc[int((data.tail(1).index.item()+1)*perc):, len(data.axes[1])-1:].values
    return X, X_rest, y, y_rest
