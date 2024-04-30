import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

def get_data() -> pd.DataFrame:
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("aerogerador.dat", names=["Vel", "Pot"], sep="\t")

def plot(data: pd.DataFrame) -> None:
    Vel = (data.iloc[:, 0].min(), data.iloc[:, 0].max())
    Pot = (data.iloc[:, 1].min(), data.iloc[:, 1].max())
    realm = [Vel, Pot]
    x_axis = np.linspace(realm[0][0], realm[0][1], 1000)
    y_axis = np.linspace(realm[1][0], realm[1][1], 1000)
    plt.plot(x_axis, y_axis)
    plt.grid()
    plt.suptitle('Espalhamento dos dados', fontsize = 12)
    plt.title("Aerogerador")
    plt.xlabel('Velocidade')
    plt.ylabel('Potência')
    for i in range(data.tail(1).index.item() +1):
        plt.scatter(data.at[i, 'Vel'], data.at[i, 'Pot'], color='green', s=30, marker='o', linewidth=1, edgecolors="black")
    plt.show()

def run() -> None:
    print(get_data())

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        plot(get_data())
        run()
finally:
    close()
