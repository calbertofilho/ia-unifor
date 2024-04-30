import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

def get_data() -> pd.DataFrame:
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("EMGDataset.csv", names=["Supercílio", "Zigomático", "Rótulo"], sep=",")

def printProgressBar(value, label):
    n_bar = 40 # tamanho da barra
    max = 100
    j = value / max
    sys.stdout.write('\r')
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1 - j))
    sys.stdout.write(f"{label.ljust(10)} | [{bar:{n_bar}s}] {int(100 * j)}% ")
    sys.stdout.flush()

def plot(data) -> None:
    Sensor1 = (data.iloc[:, 0].min(), data.iloc[:, 0].max())
    Sensor2 = (data.iloc[:, 1].min(), data.iloc[:, 1].max())
    realm = [Sensor1, Sensor2]
    x_axis = np.linspace(realm[0][0], realm[0][1], 1000)
    y_axis = np.linspace(realm[1][0], realm[1][1], 1000)
    plt.plot(x_axis, y_axis)
    plt.grid()
    plt.suptitle('Espalhamento dos dados', fontsize = 12)
    plt.title("Eletromiografia")
    plt.xlabel('Sensor 1: Supercílio')
    plt.ylabel('Sensor 2: Zigomático')
    for i in range(data.tail(1).index.item() +1):
        plt.scatter(data.at[i, 'Supercílio'], data.at[i, 'Zigomático'], color='blue', s=30, marker='o', linewidth=1, edgecolors="black")
        printProgressBar((i / (data.tail(1).index.item()+1)) * 100, 'Carregando dados')
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
