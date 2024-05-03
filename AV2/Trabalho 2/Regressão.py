import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

TITLE = 'Espalhamento dos dados'
SUBTITLE = 'Aerogerador'
X_LABEL = 'Velocidade do vento'
Y_LABEL = 'Potência gerada'
FIGURE_SIZE = (7, 5)

def get_data() -> pd.DataFrame:
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("aerogerador.dat", names=["Vel", "Pot"], sep="\t")

def preview_data(data: pd.DataFrame) -> None:
    plt.figure(figsize=FIGURE_SIZE)
    plt.suptitle(TITLE, fontsize = 16)
    plt.title(SUBTITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.grid()
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], color='blue', s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
    plt.legend(["Dados"], fancybox=True, framealpha=1, shadow=True, borderpad=1, loc="best")
    plt.show()

def run() -> None:
    # coleta dos dados
    df_embaralhado = get_data()
    # definição da quantidade de rodadas
    for _ in range(1):
        # embaralhamento dos dados
        df_embaralhado = df_embaralhado.sample(frac=1).reset_index(drop=True)
        # fatiamento dos dados: 80% ↔ 20%
        percentual = .8
        x1_treino = df_embaralhado.iloc[:int((df_embaralhado.tail(1).index.item()+1)*percentual), 0].values
        x1_treino.shape = (len(x1_treino), 1)
        x2_treino = df_embaralhado.iloc[:int((df_embaralhado.tail(1).index.item()+1)*percentual), 1].values
        x2_treino.shape = (len(x2_treino), 1)
        x1_teste = df_embaralhado.iloc[int((df_embaralhado.tail(1).index.item()+1)*percentual):(df_embaralhado.tail(1).index.item()+1), 0].values
        x1_teste.shape = (len(x1_teste), 1)
        x2_teste = df_embaralhado.iloc[int((df_embaralhado.tail(1).index.item()+1)*percentual):(df_embaralhado.tail(1).index.item()+1), 1].values
        x2_teste.shape = (len(x2_teste), 1)
        # calculo do MQO
        X = np.concatenate((np.ones((len(x1_treino), 1)), x1_treino), axis=1)
        B = np.linalg.pinv(X.T @ X) @ X.T @ x2_treino
        x_axis = np.linspace(x1_treino.min(), x1_treino.max(), len(x1_treino))
        x_axis.shape = (len(x_axis), 1)
        print(x_axis.shape)
        ones = np.ones((len(x_axis), 1))
        X_new = np.concatenate((ones, x_axis), axis=1)
        Y = X_new @ B
    # definição do gráfico
    plt.figure(figsize=FIGURE_SIZE)
    plt.suptitle(TITLE, fontsize = 16)
    plt.title(SUBTITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.grid()
    # plotagem dos pontos do grupo de treino
    plt.scatter(x1_treino, x2_treino, color='green', s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
    # plotagem dos pontos do grupo de teste
    plt.scatter(x1_teste, x2_teste, color='orange', s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
    # plotagem da linha
    plt.plot(x_axis, Y, color='red', linewidth=0.4)
    plt.legend(["Grupo de treino", "Grupo de teste", "Melhor reta"], fancybox=True, framealpha=1, shadow=True, borderpad=1, loc="best")
    plt.show()

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        preview_data(get_data())
        run()
finally:
    close()
