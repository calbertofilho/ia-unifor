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
COLORS = {
    1.0: 'blue',
    2.0: 'green',
    3.0: 'red',
    4.0: 'cyan',
    5.0: 'magenta'
}
ROTULOS = {
    1.0: [1, 0, 0, 0, 0],
    2.0: [0, 1, 0, 0, 0],
    3.0: [0, 0, 1, 0, 0],
    4.0: [0, 0, 0, 1, 0],
    5.0: [0, 0, 0, 0, 1]
}

def load_full_data() -> pd.DataFrame:
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("EMGDataset.csv", names=["Supercílio", "Zigomático", "Rótulo"], sep=",")

def check_dirs() -> None:
    # verifica e cria os diretorios necessarios para salvar os arquivos
    if not os.path.exists('AV2/Trabalho 2/Classificação'):
        os.makedirs('AV2/Trabalho 2/Classificação')

def preview_data(data: pd.DataFrame, savefig: bool) -> None:
    plt.figure(figsize=FIGURE_SIZE)
    plt.suptitle(TITLE, fontsize = 16)
    plt.title(SUBTITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)
    plt.grid()
    color_list = [COLORS[group] for group in data['Rótulo']]
    plt.scatter(data['Supercílio'], data['Zigomático'], color=color_list, s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
    legend_handles = [
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=COLORS[1.0], linewidth=0.4, alpha=0.6, label="Neutro"),
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=COLORS[2.0], linewidth=0.4, alpha=0.6, label="Sorriso"),
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=COLORS[3.0], linewidth=0.4, alpha=0.6, label="Sobrancelhas levantadas"),
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=COLORS[4.0], linewidth=0.4, alpha=0.6, label="Surpreso"),
        Line2D([0], [0], color='none', marker='o', markersize=8, markerfacecolor=COLORS[5.0], linewidth=0.4, alpha=0.6, label="Rabugento")
    ]
    plt.legend(title="Expressões", handles=legend_handles, fancybox=True, framealpha=1, shadow=True, borderpad=1, loc="best")
    if savefig:
        nome_arquivo = 'espalhamento'
        # cria a imagem .png estática do gráfico
        plt.savefig('Classificação/%s.png' % nome_arquivo)
    plt.show()

def calc_mqo(data: pd.DataFrame, perc_fatiamento: float) -> float:
    # fatiamento dos dados
    mqo_corte=int((data.tail(1).index.item()+1)*perc_fatiamento)
    mqo_x_treino = data.iloc[:mqo_corte, :2]
    mqo_X_treino = np.concatenate((np.ones((len(mqo_x_treino), 1)), mqo_x_treino), axis=1)
    mqo_y_treino = np.vstack(data.iloc[:mqo_corte, 2:]["Rótulo"].map(ROTULOS))
    mqo_x_teste = data.iloc[mqo_corte:, :2]
    mqo_X_teste = np.concatenate((np.ones((len(mqo_x_teste), 1)), mqo_x_teste), axis=1)
    mqo_y_teste = data.iloc[mqo_corte:, 2:]
    # cálculo do MQO
    mqo_w = np.linalg.pinv(mqo_X_treino.T @ mqo_X_treino) @ mqo_X_treino.T @ mqo_y_treino
    mqo_y_predito = mqo_X_teste @ mqo_w
    return (sum(mqo_y_teste['Rótulo'] == mqo_y_predito.argmax(axis=1)+1) / len(mqo_y_predito)) #ACURÁCIA

def calc_tikhonov(data: pd.DataFrame, perc_fatiamento: float) -> np.ndarray:
    # fatiamento dos dados
    tik_corte=int((data.tail(1).index.item()+1)*perc_fatiamento)
    tik_x_treino = data.iloc[:tik_corte, :2]
    tik_X_treino = np.concatenate((np.ones((len(tik_x_treino), 1)), tik_x_treino), axis=1)
    tik_y_treino = np.vstack(data.iloc[:tik_corte, 2:]["Rótulo"].map(ROTULOS))
    tik_x_teste = data.iloc[tik_corte:, :2]
    tik_X_teste = np.concatenate((np.ones((len(tik_x_teste), 1)), tik_x_teste), axis=1)
    tik_y_teste = data.iloc[tik_corte:, 2:]
    # cálculo do MQO Regularizado (Tikhonov)
    tik_I = np.identity(len(tik_X_treino[0])) # I₍ₚ․ₚ₎
    res = np.empty(10)
    for lamb in range(1, 11):
        tik_w = np.linalg.pinv((tik_X_treino.T @ tik_X_treino) + (tik_I * (lamb / 10))) @ tik_X_treino.T @ tik_y_treino
        tik_y_predito = tik_X_teste @ tik_w
        res[lamb-1] = (sum(tik_y_teste['Rótulo'] == tik_y_predito.argmax(axis=1)+1) / len(tik_y_predito)) #ACURÁCIA
    return res

def printProgressBar(value: float, label: str) -> None:
    n_bar = 40 # tamanho da barra
    max = 100
    j = value / max
    bar = '█' * int(n_bar * j)
    bar = bar + '-' * int(n_bar * (1 - j))
    sys.stdout.write('\r')
    sys.stdout.write(f"{label.ljust(10)} | [{bar:{n_bar}s}] {int(100 * j)}% ")
    sys.stdout.flush()

def run(save:bool) -> None:
    # coleta dos dados
    data = load_full_data()
    # definição do fatiamento dos dados: 80% ↔ 20%
    percentual = .8
    # definição da quantidade de rodadas
    rodadas = 100
    nome_arquivo = 'resultado'
    resultados = np.empty([rodadas, 11])
    for rodada in range(rodadas):
        printProgressBar((rodada / rodadas) * 100, 'Calculando...')
        # embaralhamento dos dados
        data = data.sample(frac=1).reset_index(drop=True)
        resultados[rodada] = np.append([ calc_mqo(data, percentual) ], [ calc_tikhonov(data, percentual) ])
    printProgressBar(100, 'Concluído !!!')
    msg = "\n\nAcurácias\n---------\n\n"
    msg += (f"        M Q O\nMenor valor: {resultados[:, :1].min():.5f}\nMaior valor: {resultados[:, :1].max():.5f}\nMédia: {resultados[:, :1].mean():.5f}\nDesvio padrão: {resultados[:, :1].std():.5f}\nModas:\n{resultados[:4, :1]}\n ... {resultados[:, :1].shape}\n\n")
    media_acuracias = []
    for i in range(1, 11, 1):
        media_acuracias.append(resultados[:, i].mean())
    pos = media_acuracias.index(max(media_acuracias)) +1 #Maior acurácia
    msg += (f"      Tikhonov\nMelhor lambda: {pos / 10}\nMenor valor: {resultados[:, pos].min():.5f}\nMaior valor: {resultados[:, pos].max():.5f}\nMédia: {resultados[:, pos].mean():.5f}\nDesvio padrão: {resultados[:, pos].std():.5f}\nModas:\n{resultados[:4, 1:]}\n ... {resultados[:, 1:].shape}\n\n")
    if save:
        # cria arquivo de texto com os resultados dos cálculos
        with open('Classificação/%s.txt' % nome_arquivo, 'w') as arquivo:
            arquivo.write(msg)
        pd.DataFrame(data=resultados[:, :1], columns=['MQO']).to_csv('Classificação/acuracia_mqo.csv', index=False)
        pd.DataFrame(data=resultados[:, 1:], columns=['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']).to_csv('Classificação/acuracia_tikhonov.csv', index=False)
    print(msg)

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        check_dirs()
        preview_data(load_full_data(), savefig=True)
        run(save=True)
finally:
    close()
