import sys
import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot
from classificadores.mlp import MultilayerPerceptron as MLP
from classificadores.adaline import Adaline as Ada
from classificadores.perceptron import Perceptron as Perc
from classificadores.percp_new import Perceptron as Percep
from utils.manipulation import clearScreen, loadData, shuffleData, partitionData
from utils.progress import printProgressBar, printAnimatedBar

def new(inputData: pd.DataFrame) -> None:
    perceptron = Percep(tx_aprendizado=0.001, n_iteracoes=10)                   # Inicia o classificador com a taxa de aprendizado e o número de épocas para iterações
    data = shuffleData(inputData)                                               # Embaralha os dados
    X_trn, X_tst, y_trn, y_tst = partitionData(data, 0.8)                       # Particiona os dados no percentual proposto
    perceptron.treinamento(X_trn, y_trn)                                        # Treina o classificador com os dados separados para treinamento
    print(perceptron.predicao(X_tst))
    print(y_tst)

def run(inputData: pd.DataFrame) -> None:
    rodada = 0                                                                  # Contador das rodadas
    rodadas = 100                                                               # Número máximo de rodadas
    dados_rodada = []
    calculos = []
    while (rodada < rodadas):
        printProgressBar((rodada / rodadas) * 100, 'Calculando...')
        ppn = Perc(tx_apr=0.001, n_epochs=100)                                  # Inicia o classificador com a taxa de aprendizado e o número de épocas para iterações
        data = shuffleData(inputData)                                           # Embaralha os dados
        X_trn, X_tst, y_trn, y_tst = partitionData(data, 0.8)                   # Particiona os dados no percentual proposto
        pesos = ppn.treinamento(X=X_trn, y=y_trn)                               # Calcula os pesos da fase de treinamento do classificador
        a, s, e, q = ppn.predicao(pesos, X_tst, y_tst)                          # Calcula os valores de acuracia, sensibilidade, especificidade e eqm para a fase de teste
        dados_rodada.append(
            {
                "rodada": rodada+1,
                "acuracia": a,
                "sensibilidade": s,
                "especificidade": e,
                "eqm": q,
                "pesos": pesos,
                "matriz_confusao": ppn.getMatrizConfusao()
            }
        )
        rodada += 1                                                             # Incrementa o contador de rodadas
        ppn = data = X_trn = X_tst = y_trn = y_tst = pesos = None               # Reseta o classificador e os dados de entrada
        a = s = e = q = 0                                                       # Reseta os valores calculados
    printProgressBar(100, 'Concluído !!!')
    estatisticas = pd.DataFrame(dados_rodada)
    calculos.append(
        {
            "Perceptron simples":
                {
                    "eqm": ["{:.4f}".format(num.mean(estatisticas.iloc[:]["eqm"])), "{:.4f}".format(num.median(estatisticas.iloc[:]["eqm"])), "{:.4f}".format(num.min(estatisticas.iloc[:]["eqm"])), "{:.4f}".format(num.max(estatisticas.iloc[:]["eqm"])), "{:.4f}".format(num.std(estatisticas.iloc[:]["eqm"]))],
                    "acuracia": ["{:.4f}".format(num.mean(estatisticas.iloc[:]["acuracia"])), "{:.4f}".format(num.median(estatisticas.iloc[:]["acuracia"])), "{:.4f}".format(num.min(estatisticas.iloc[:]["acuracia"])), "{:.4f}".format(num.max(estatisticas.iloc[:]["acuracia"])), "{:.4f}".format(num.std(estatisticas.iloc[:]["acuracia"]))],
                    "especificidade": ["{:.4f}".format(num.mean(estatisticas.iloc[:]["especificidade"])), "{:.4f}".format(num.median(estatisticas.iloc[:]["especificidade"])), "{:.4f}".format(num.min(estatisticas.iloc[:]["especificidade"])), "{:.4f}".format(num.max(estatisticas.iloc[:]["especificidade"])), "{:.4f}".format(num.std(estatisticas.iloc[:]["especificidade"]))],
                    "sensibilidade": ["{:.4f}".format(num.mean(estatisticas.iloc[:]["sensibilidade"])), "{:.4f}".format(num.median(estatisticas.iloc[:]["sensibilidade"])), "{:.4f}".format(num.min(estatisticas.iloc[:]["sensibilidade"])), "{:.4f}".format(num.max(estatisticas.iloc[:]["sensibilidade"])), "{:.4f}".format(num.std(estatisticas.iloc[:]["sensibilidade"]))]
                }
        }
    )
    resultados = pd.DataFrame(calculos)
    print("\n\n", resultados.keys().values[0])
    print(pd.DataFrame(data=resultados.iloc[:]['Perceptron simples'][0], index=["media", "mediana", "minimo", "maximo", "d.padrao"]).T)
    print()

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        espiral = loadData(file_name="spiral.csv", columns=["x1", "x2", "y",], separator=",", ignore_header=False)
        aerogerador = loadData("aerogerador.dat", ["Vel", "Pot"], "\t", False)
        red_wine = loadData("winequality-red.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        white_wine = loadData("winequality-white.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        clearScreen()
        new(inputData=espiral)
        # run(inputData=espiral)
        # run(inputData=aerogerador)
        # run(inputData=red_wine)
        # run(inputData=white_wine)
finally:
    close()
