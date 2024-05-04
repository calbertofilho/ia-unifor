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
    resultados = np.empty(1000)
    resultados.shape = (1000, 1)
    # definição da quantidade de rodadas
    for rodada in range(1000):
        # embaralhamento dos dados
        df_embaralhado = df_embaralhado.sample(frac=1).reset_index(drop=True)
        # fatiamento dos dados: 80% ↔ 20%
        percentual = .8
        x_treino = df_embaralhado.iloc[:int((df_embaralhado.tail(1).index.item()+1)*percentual), 0].values
        x_treino.shape = (len(x_treino), 1)
        y_treino = df_embaralhado.iloc[:int((df_embaralhado.tail(1).index.item()+1)*percentual), 1].values
        y_treino.shape = (len(y_treino), 1)
        xy_treino = (x_treino * y_treino)
        xy_treino.shape = (len(y_treino), 1)
        x_teste = df_embaralhado.iloc[int((df_embaralhado.tail(1).index.item()+1)*percentual):, 0].values
        x_teste.shape = (len(x_teste), 1)
        y_teste = df_embaralhado.iloc[int((df_embaralhado.tail(1).index.item()+1)*percentual):, 1].values
        y_teste.shape = (len(y_teste), 1)
        xy_teste = (x_teste * y_teste)
        xy_teste.shape = (len(y_teste), 1)
        # regressão linear univariada   →   y = a·x + b
        # xy = x · y
        # a = ((len(x) · xy.sum()) - (x.sum() · y.sum())) / ((len(x) · (x**2).sum()) - x.sum()**2)
        a = ((len(x_treino) * xy_treino.sum()) - (x_treino.sum() * y_treino.sum())) / ((len(x_treino) * (np.square(x_treino)).sum()) - np.square(x_treino.sum()))
        # b = y.mean() - (a · x.mean())
        b = y_treino.mean() - (a * x_treino.mean())
        y_predito = np.empty(len(x_teste))
        y_predito.shape = (len(y_predito), 1)
        for i in range(len(x_teste)):
            y_predito[i] = (a * x_teste[i]) + b
        EQM = np.square(np.subtract(y_teste, y_predito)).mean()
        resultados[rodada] = EQM
    print(f"\nErro Quadrático Médio\n---------------------\n        (MQO)\nMédia: {resultados.mean():.2f}\nDesvio padrão: {resultados.std():.2f}\nMenor valor: {resultados.min():.2f}\nMaior valor: {resultados.max():.2f}\n")


    # # definição do gráfico
    # plt.figure(figsize=FIGURE_SIZE)
    # plt.suptitle(TITLE, fontsize = 16)
    # plt.title(SUBTITLE)
    # plt.xlabel(X_LABEL)
    # plt.ylabel(Y_LABEL)
    # plt.grid()
    # # plotagem dos pontos do grupo de treino
    # plt.scatter(x_treino, y_treino, color='green', s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
    # # plotagem dos pontos do grupo de teste
    # plt.scatter(x_teste, y_teste, color='orange', s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
    # # plotagem da linha
    # # plt.plot(x_axis, Y, color='red', linewidth=0.4)
    # plt.legend(["Grupo de treino", "Grupo de teste", "Melhor reta"], fancybox=True, framealpha=1, shadow=True, borderpad=1, loc="best")
    # plt.show()

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        preview_data(get_data())
        run()
finally:
    close()
