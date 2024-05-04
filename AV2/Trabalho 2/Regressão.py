import os
import sys
import numpy as np
import pandas as pd
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
    obs_y_prd = np.empty(len(obs_x_tst))
    obs_y_prd.shape = (len(obs_y_prd), 1)
    # Valores observados   →   y = a·x + b
    # xy = x · y
    # a = ((len(x) · xy.sum()) - (x.sum() · y.sum())) / ((len(x) · (x**2).sum()) - x.sum()**2)
    # b = y.mean() - (a · x.mean())
    a = ((len(obs_x_trn) * obs_xy_trn.sum()) - (obs_x_trn.sum() * obs_y_trn.sum())) / ((len(obs_x_trn) * (np.square(obs_x_trn)).sum()) - np.square(obs_x_trn.sum()))
    b = obs_y_trn.mean() - (a * obs_x_trn.mean())
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
    mqo_X_trn = np.concatenate((np.ones((len(mqo_x_trn), 1)), mqo_x_trn), axis=1)
    mqo_X_tst = np.concatenate((np.ones((len(mqo_x_tst), 1)), mqo_x_tst), axis=1)
    # MQO   →   y = X_teste · W
    # w = (X_treino.T · X_treino)^-1 · X_treino.T · y_treino
    mqo_w = np.linalg.pinv(mqo_X_trn.T @ mqo_X_trn) @ mqo_X_trn.T @ mqo_y_trn
    mqo_Y_prd = mqo_X_tst @ mqo_w
    return np.square(np.subtract(mqo_y_tst, mqo_Y_prd)).mean() #EQM

def calc_tikhonov(data, perc_fatiamento) -> np.ndarray:
    tik_x_trn = data.iloc[:int((data.tail(1).index.item()+1)*perc_fatiamento), 0].values
    tik_y_trn = data.iloc[:int((data.tail(1).index.item()+1)*perc_fatiamento), 1].values
    tik_x_trn.shape = tik_y_trn.shape = (len(tik_x_trn), 1)
    tik_x_tst = data.iloc[int((data.tail(1).index.item()+1)*perc_fatiamento):, 0].values
    tik_y_tst = data.iloc[int((data.tail(1).index.item()+1)*perc_fatiamento):, 1].values
    tik_x_tst.shape = tik_y_tst.shape = (len(tik_x_tst), 1)
    tik_X_trn = np.concatenate((np.ones((len(tik_x_trn), 1)), tik_x_trn), axis=1)
    tik_I = np.identity((len(tik_x_trn)))
    tik_X_tst = np.concatenate((np.ones((len(tik_x_tst), 1)), tik_x_tst), axis=1)
    res = np.empty(10)
    # Tikhonov   →   y = X_teste · W
    # 0 < ⅄ <= 1
    # w = ((X_treino.T · X_treino) + ⅄I)^-1 · X_treino.T · y_treino
    for lamb in range(1, 11):
        lambI = (lamb * tik_I)
        tik_w = np.linalg.pinv((tik_X_trn.T @ tik_X_trn) + lamb) @ tik_X_trn.T @ tik_y_trn       # <- ERRO na soma do produto de lambda pela Matriz Identidade (lambI)
        tik_Y_prd = tik_X_tst @ tik_w
        res[lamb-1] = np.square(np.subtract(tik_y_tst, tik_Y_prd)).mean()
    return res

def printProgressBar(value, label):
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
    df_embaralhado = get_data()
    percentual = .8
    rodadas = 1000
    resultados = np.empty([rodadas, 12])
    # fatiamento dos dados: 80% ↔ 20%
    # definição da quantidade de rodadas
    for rodada in range(rodadas):
        # embaralhamento dos dados
        printProgressBar((rodada / rodadas) * 100, 'Calculando...')
        df_embaralhado = df_embaralhado.sample(frac=1).reset_index(drop=True)
        resultados[rodada] = np.append([ calc_valorobs(df_embaralhado, percentual), calc_mqo(df_embaralhado, percentual) ], [ calc_tikhonov(df_embaralhado, percentual) ])
    printProgressBar(100, 'Concluído !!!')
    print("\n\nErro Quadrático Médio\n---------------------\n")
    print(f" Valores  observados\nMenor valor: {resultados[:, 0].min():.2f}\nMaior valor: {resultados[:, 0].max():.2f}\nMédia: {resultados[:, 0].mean():.2f}\nDesvio padrão: {resultados[:, 0].std():.2f}\n")
    print(f"        M Q O\nMenor valor: {resultados[:, 1].min():.2f}\nMaior valor: {resultados[:, 1].max():.2f}\nMédia: {resultados[:, 1].mean():.2f}\nDesvio padrão: {resultados[:, 1].std():.2f}\n")
    eqm_minimo = []
    for i in range(2, 12, 1):
        eqm_minimo.append(resultados[:, i].mean())
    pos = eqm_minimo.index(min(eqm_minimo))
    print(f"      Tikhonov\nMelhor lambda: {(pos + 1) / 10}\nMenor valor: {resultados[:, pos].min():.2f}\nMaior valor: {resultados[:, pos].max():.2f}\nMédia: {resultados[:, pos].mean():.2f}\nDesvio padrão: {resultados[:, pos].std():.2f}\n")

def plot_mqo(data):
        # embaralhamento dos dados
        data = data.sample(frac=1).reset_index(drop=True)
        # fatiamento dos dados: 80% ↔ 20%
        percentual = .8
        x_treino = data.iloc[:int((data.tail(1).index.item()+1)*percentual), 0].values
        x_treino.shape = (len(x_treino), 1)
        y_treino = data.iloc[:int((data.tail(1).index.item()+1)*percentual), 1].values
        y_treino.shape = (len(y_treino), 1)
        x_teste = data.iloc[int((data.tail(1).index.item()+1)*percentual):, 0].values
        x_teste.shape = (len(x_teste), 1)
        y_teste = data.iloc[int((data.tail(1).index.item()+1)*percentual):, 1].values
        y_teste.shape = (len(y_teste), 1)
        # calculo do MQO
        X = np.concatenate((np.ones((len(x_treino), 1)), x_treino), axis=1)
        B = np.linalg.pinv(X.T @ X) @ X.T @ y_treino
        x_axis = np.linspace(data.iloc[:, 0].min(), data.iloc[:, 0].max(), len(data.iloc[:, 0]))
        x_axis.shape = (len(x_axis), 1)
        ones = np.ones((len(x_axis), 1))
        X_new = np.concatenate((ones, x_axis), axis=1)
        Y = X_new @ B
        # definição do gráfico
        plt.figure(figsize=FIGURE_SIZE)
        plt.suptitle("Predição da melhor reta [MQO]", fontsize = 16)
        plt.title(SUBTITLE)
        plt.xlabel(X_LABEL)
        plt.ylabel(Y_LABEL)
        plt.grid()
        # plotagem dos pontos do grupo de treino
        plt.scatter(x_treino, y_treino, color='green', s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
        # plotagem dos pontos do grupo de teste
        plt.scatter(x_teste, y_teste, color='orange', s=40, marker='o', linewidth=0.4, edgecolors="black", alpha=0.6)
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
        plot_mqo(get_data())
finally:
    close()
