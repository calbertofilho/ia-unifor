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
    # coleta dos dados
    data = get_data()
    x = (data['Supercílio'].values, data['Zigomático'].values)
    y = data['Rótulo'].values
    # definição da quantidade de rodadas
    for _ in range(1):
        # embaralhamento dos dados
        data = data.sample(frac=1).reset_index(drop=True)
        # definição do fatiamento dos dados: 80% ↔ 20%
        percentual = .8
        # fatiamento dos dados
        x_treino = (data.iloc[:int((data.tail(1).index.item()+1)*percentual), 0].values, data.iloc[:int((data.tail(1).index.item()+1)*percentual), 1].values)
        y_treino = data.iloc[:int((data.tail(1).index.item()+1)*percentual), 2].values
        X_treino = np.concatenate((np.ones((len(x_treino), 1)), x_treino), axis=1)
        x_teste = (data.iloc[int((data.tail(1).index.item()+1)*percentual):, 0].values, data.iloc[:int((data.tail(1).index.item()+1)*percentual), 1].values)
        y_teste = data.iloc[int((data.tail(1).index.item()+1)*percentual):, 2].values
        # regressão linear multivariada
        # MQO   →   y = X_teste · W
        # w = (X_treino.T · X_treino)^-1 · X_treino.T · y_treino
        w = np.linalg.pinv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
        X_teste = np.concatenate((np.ones((len(x_teste), 1)), x_teste), axis=1)
        y_predito = X_teste @ w
        print(y_teste[135], y_predito[135])
        # Tikhonov   →   y = X_teste · W
        # 0 < ⅄ <= 1
        # w = ((X_treino.T · X_treino) + ⅄I)^-1 · X_treino.T · y_treino
        # I = np.identity(len(x_teste))
        # w = np.linalg.pinv((X_treino.T @ X_treino) + (0.1 * I)) @ X_treino.T @ y_treino

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        preview_data(get_data())
        # run()
finally:
    close()
