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

def run() -> None:
    df = df_shuffled = get_data()
    # embaralhamento dos dados
    df_shuffled = df_shuffled.sample(frac=1).reset_index(drop=True)
    # fatiamento dos dados: 80% ↔ 20%
    percentual = .8
    x_treino = df.iloc[:int((df_shuffled.tail(1).index.item()+1)*percentual), 0].values
    x_treino.shape = (len(x_treino), 1)
    y_treino = df.iloc[:int((df_shuffled.tail(1).index.item()+1)*percentual), 1].values
    y_treino.shape = (len(y_treino), 1)
    x_teste = df.iloc[int((df_shuffled.tail(1).index.item()+1)*percentual):(df.tail(1).index.item()+1), 0].values
    x_teste.shape = (len(x_teste), 1)
    y_teste = df.iloc[int((df_shuffled.tail(1).index.item()+1)*percentual):(df.tail(1).index.item()+1), 1].values
    y_teste.shape = (len(y_teste), 1)
    # implementação MQO
    plt.suptitle('Espalhamento dos dados', fontsize = 16)
    plt.title("Aerogerador")
    plt.xlabel('Velocidade do vento')
    plt.ylabel('Potência gerada')
    plt.grid()
    # x = df["Vel"].values
    # x.shape = (len(x), 1)
    # y = df["Pot"].values
    # y.shape = (len(y), 1)
    plt.scatter(x_treino, y_treino, color='green', s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
    X = np.concatenate((np.ones((len(x_treino), 1)), x_treino), axis=1)
    B = np.linalg.pinv(X.T @ X) @ X.T @ y_treino
    x_axis = np.linspace(x_treino.min(), x_treino.max(), 100)
    x_axis.shape = (len(x_axis), 1)
    ones = np.ones((len(x_axis), 1))
    X_new = np.concatenate((ones, x_axis), axis=1)
    Y = X_new @ B
    plt.plot(x_axis, Y, color='red')
    plt.show()

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        run()
finally:
    close()
