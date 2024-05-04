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

def calc_valorobs(data, perc_fatiamento) -> float:
    obs_x_trn = data.iloc[:int((data.tail(1).index.item()+1)*perc_fatiamento), 0].values
    obs_y_trn = data.iloc[:int((data.tail(1).index.item()+1)*perc_fatiamento), 1].values
    obs_xy_trn = (obs_x_trn * obs_y_trn)
    obs_x_trn.shape = obs_y_trn.shape = obs_xy_trn.shape = (len(obs_x_trn), 1)
    obs_x_tst = data.iloc[int((data.tail(1).index.item()+1)*perc_fatiamento):, 0].values
    obs_y_tst = data.iloc[int((data.tail(1).index.item()+1)*perc_fatiamento):, 1].values
    obs_xy_tst = (obs_x_tst * obs_y_tst)
    obs_x_tst.shape = obs_y_tst.shape = obs_xy_tst.shape = (len(obs_x_tst), 1)
    # Valores observados   →   y = a·x + b
    # xy = x · y
    # a = ((len(x) · xy.sum()) - (x.sum() · y.sum())) / ((len(x) · (x**2).sum()) - x.sum()**2)
    # b = y.mean() - (a · x.mean())
    a = ((len(obs_x_trn) * obs_xy_trn.sum()) - (obs_x_trn.sum() * obs_y_trn.sum())) / ((len(obs_x_trn) * (np.square(obs_x_trn)).sum()) - np.square(obs_x_trn.sum()))
    b = obs_y_trn.mean() - (a * obs_x_trn.mean())
    obs_y_prd = np.empty(len(obs_x_tst))
    obs_y_prd.shape = (len(obs_y_prd), 1)
    for i in range(len(obs_x_tst)):
        obs_y_prd[i] = (a * obs_x_tst[i]) + b
    return np.square(np.subtract(obs_y_tst, obs_y_prd)).mean() #EQM

def calc_mqo(data, perc_fatiamento) -> float:
    mqo_x_trn = data.iloc[:int((data.tail(1).index.item()+1)*perc_fatiamento), 0].values
    mqo_y_trn = data.iloc[:int((data.tail(1).index.item()+1)*perc_fatiamento), 1].values
    mqo_x_trn.shape = mqo_y_trn.shape = (len(mqo_x_trn), 1)
    mqo_x_tst = data.iloc[int((data.tail(1).index.item()+1)*perc_fatiamento):, 0].values
    mqo_y_tst = data.iloc[int((data.tail(1).index.item()+1)*perc_fatiamento):, 1].values
    mqo_x_tst.shape = mqo_y_tst.shape = (len(mqo_x_tst), 1)
    # MQO   →   y = X_teste · W
    # w = (X_treino.T · X_treino)^-1 · X_treino.T · y_treino
    mqo_X_trn = np.concatenate((np.ones((len(mqo_x_trn), 1)), mqo_x_trn), axis=1)
    mqo_w = np.linalg.pinv(mqo_X_trn.T @ mqo_X_trn) @ mqo_X_trn.T @ mqo_y_trn
    mqo_X_tst = np.concatenate((np.ones((len(mqo_x_tst), 1)), mqo_x_tst), axis=1)
    mqo_Y_prd = mqo_X_tst @ mqo_w
    return np.square(np.subtract(mqo_y_tst, mqo_Y_prd)).mean() #EQM

def calc_tikhonov(data, perc_fatiamento, lamb) -> float:
    tik_x_trn = data.iloc[:int((data.tail(1).index.item()+1)*perc_fatiamento), 0].values
    tik_y_trn = data.iloc[:int((data.tail(1).index.item()+1)*perc_fatiamento), 1].values
    tik_x_trn.shape = tik_y_trn.shape = (len(tik_x_trn), 1)
    tik_x_tst = data.iloc[int((data.tail(1).index.item()+1)*perc_fatiamento):, 0].values
    tik_y_tst = data.iloc[int((data.tail(1).index.item()+1)*perc_fatiamento):, 1].values
    tik_x_tst.shape = tik_y_tst.shape = (len(tik_x_tst), 1)
    # Tikhonov   →   y = X_teste · W
    # 0 < ⅄ <= 1
    # w = ((X_treino.T · X_treino) + ⅄I)^-1 · X_treino.T · y_treino
    tik_X_trn = np.concatenate((np.ones((len(tik_x_trn), 1)), tik_x_trn), axis=1)
    tik_I = np.identity((len(tik_x_trn)))
    tik_w = np.linalg.pinv((tik_X_trn.T @ tik_X_trn) + (lamb * tik_I)) @ tik_X_trn.T @ tik_y_trn
    tik_X_tst = np.concatenate((np.ones((len(tik_x_tst), 1)), tik_x_tst), axis=1)
    tik_Y_prd = tik_X_tst @ tik_w
    return np.square(np.subtract(tik_y_tst, tik_Y_prd)).mean() #EQM

def run() -> None:
    # coleta dos dados
    df_embaralhado = get_data()
    percentual = .8
    rodadas = 1000
    resultados = np.empty(rodadas)
    # resultados.shape = (rodadas, 3)
    # fatiamento dos dados: 80% ↔ 20%
    # definição da quantidade de rodadas
    for rodada in range(rodadas):
        # embaralhamento dos dados
        df_embaralhado = df_embaralhado.sample(frac=1).reset_index(drop=True)
        # print(calc_valorobs(df_embaralhado, percentual))
        # print(calc_mqo(df_embaralhado, percentual))
        for lamb in range(1, 11):
            print(calc_tikhonov(df_embaralhado, percentual, lamb/10))
    # print("Erro Quadrático Médio\n---------------------\n")
    # print(f"Valores  observados\nMenor valor: {res_obs.min():.2f}\nMaior valor: {res_obs.max():.2f}\nMédia: {res_obs.mean():.2f}\nDesvio padrão: {res_obs.std():.2f}\n")
    # print(f"        M Q O\nMenor valor: {res_mqo.min():.2f}\nMaior valor: {res_mqo.max():.2f}\nMédia: {res_mqo.mean():.2f}\nDesvio padrão: {res_mqo.std():.2f}\n")
    # print(f"      Tikhonov\n")


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
        # preview_data(get_data())
        run()
finally:
    close()
