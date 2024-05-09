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

def printProgressBar(value: float, label: str) -> None:
    n_bar = 40 # tamanho da barra
    max = 100
    j = value / max
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1 - j))
    sys.stdout.write('\r')
    sys.stdout.write(f"{label.ljust(10)} | [{bar:{n_bar}s}] {int(100 * j)}% ")
    sys.stdout.flush()

def run() -> None:
    # coleta dos dados
    data = get_data()
    x = (data['Supercílio'].values, data['Zigomático'].values)
    y = data['Rótulo'].values
    rotulos = {
        1.0: [1, -1, -1, -1, -1],
        2.0: [-1, 1, -1, -1, -1],
        3.0: [-1, -1, 1, -1, -1],
        4.0: [-1, -1, -1, 1, -1],
        5.0: [-1, -1, -1, -1, 1]
    }
    # definição da quantidade de rodadas
    for _ in range(1000):
        # embaralhamento dos dados
        data = data.sample(frac=1).reset_index(drop=True)
        # definição do fatiamento dos dados: 80% ↔ 20%
        percentual = .8
        # fatiamento dos dados
        x_treino = data.iloc[:int((data.tail(1).index.item()+1)*percentual), :2]
        y_treino = data.iloc[:int((data.tail(1).index.item()+1)*percentual), 2:]
        X_treino = np.concatenate((np.ones((len(x_treino), 1)), x_treino), axis=1)
        x_teste = data.iloc[int((data.tail(1).index.item()+1)*percentual):, :2]
        y_teste = data.iloc[int((data.tail(1).index.item()+1)*percentual):, :2]
        I = np.identity(len(X_treino[0])) # I₍ₚ․ₚ₎
        # regressão linear multivariada
        # MQO   →   y = X_teste · W
        # W = (Xᵀ · X)⁻¹ · Xᵀ · y
        w = np.linalg.pinv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
        print(f"w {w}")
        print(rotulos[np.argmax(w) + 1])
        # W = (Y · Xᵀ) · (X · Xᵀ)⁻¹
        # w = (y_treino @ X_treino.T) @ np.linalg.pinv(X_treino @ X_treino.T)
        X_teste = np.concatenate((np.ones((len(x_teste), 1)), x_teste), axis=1)
        y_predito = X_teste @ w
        print(f"y = {y_predito}")
        # Tikhonov   →   y = X_teste · W
        # 0 < ⅄ <= 1
        # W = ((Xᵀ · X) + (⅄ · I₍ₚ․ₚ₎))⁻¹ · Xᵀ · y
        wI = np.linalg.pinv((X_treino.T @ X_treino) + (0.3 * I)) @ X_treino.T @ y_treino   #      <-- ERRO nesta linha
        print(f"wI{wI}")
        print(rotulos[np.argmax(wI) + 1])
        yI_predito = X_teste @ wI
        print(f"yI = {yI_predito}")

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        # preview_data(get_data())
        run()
finally:
    close()
