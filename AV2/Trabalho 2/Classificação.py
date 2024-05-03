import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FIGURE_SIZE = (9, 7)
TITLE = 'Espalhamento dos dados'
SUBTITLE = 'Eletromiografia'
X_LABEL = 'Sensor 1\nCorrugador do supercílio'
Y_LABEL = 'Sensor 2\nZigomático maior'

def get_data() -> pd.DataFrame:
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("EMGDataset.csv", names=["Supercílio", "Zigomático", "Rótulo"], sep=",")

def preview_data(data: pd.DataFrame) -> None:
    plt.figure(figsize=FIGURE_SIZE)
    plt.suptitle(TITLE, fontsize = 16)
    plt.title(SUBTITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.grid()
    colors = {
        1.0: 'blue',
        2.0: 'green',
        3.0: 'red',
        4.0: 'cyan',
        5.0: 'magenta'
    }
    color_list = [colors[group] for group in data['Rótulo']]
    plt.scatter(data['Supercílio'], data['Zigomático'], color=color_list, s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
    legend_handles = [
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=colors[1.0], linewidth=0.4, alpha=0.6, label="Neutro"),
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=colors[2.0], linewidth=0.4, alpha=0.6, label="Sorriso"),
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=colors[3.0], linewidth=0.4, alpha=0.6, label="Sobrancelhas levantadas"),
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=colors[4.0], linewidth=0.4, alpha=0.6, label="Surpreso"),
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=colors[5.0], linewidth=0.4, alpha=0.6, label="Rabugento")
    ]
    plt.legend(title="Expressões", handles=legend_handles, fancybox=True, framealpha=1, shadow=True, borderpad=1, loc="best")
    plt.show()

def run() -> None:
    data = get_data()
    x1 = data['Supercílio'].values
    x2 = data['Zigomático'].values
    y = data['Rótulo'].values
    for _ in range(1):
        data = data.sample(frac=1).reset_index(drop=True)
        percentual = .8
        x1_treino = data.iloc[:int((data.tail(1).index.item()+1)*percentual), 0].values
        x1_treino.shape = (len(x1_treino), 1)
        x2_treino = data.iloc[:int((data.tail(1).index.item()+1)*percentual), 1].values
        x2_treino.shape = (len(x2_treino), 1)
        y_treino = data.iloc[:int((data.tail(1).index.item()+1)*percentual), 2].values
        X = np.concatenate((np.ones((len(x1_treino), 1)), x1_treino), axis=1)
        B = np.linalg.pinv(X.T @ X) @ X.T @ x2_treino
        x_axis = np.linspace(x1_treino.min(), x1_treino.max(), len(x1_treino))
        x_axis.shape = (len(x_axis), 1)
        ones = np.ones((len(x_axis), 1))
        X_new = np.concatenate((ones, x_axis), axis=1)
        Y = X_new @ B
        # print(y_treino[:2] == Y[:])
        # print(y_treino[:])

        # x1_teste = data.iloc[int((data.tail(1).index.item()+1)*percentual):(data.tail(1).index.item()+1), 0].values
        # x1_teste.shape = (len(x1_teste), 1)
        # x2_teste = data.iloc[int((data.tail(1).index.item()+1)*percentual):(data.tail(1).index.item()+1), 1].values
        # x2_teste.shape = (len(x2_teste), 1)
        # y_teste = data.iloc[int((data.tail(1).index.item()+1)*percentual):(data.tail(1).index.item()+1), 2].values

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        preview_data(get_data())
        run()
finally:
    close()
