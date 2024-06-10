import os
import sys
import platform
import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot
from pathlib import Path
from algoritmos.mlp import MLP
from algoritmos.adaline import Adaline
from algoritmos.perceptron import Perceptron
from utils.manipulation import load_full_data, shuffleData, partitionData
from utils.progress import printProgressBar, printAnimatedBar

def run(inputData: pd.DataFrame) -> None:
    rounds = 0
    max_rounds = 100
    rodada = []
    calculos = []
    while (rounds < max_rounds):
        printProgressBar((rounds / max_rounds) * 100, 'Calculando...')
        ppn = Perceptron(tx_apr=0.001, n_epochs=100)
        data = shuffleData(inputData)
        X_trn, X_tst, y_trn, y_tst = partitionData(data, 0.8)
        ppn.treinar(X=X_trn, y=y_trn)
        a, s, e = ppn.testar(X_tst, y_tst)
        rodada.append(
            {
                "rodada": rounds+1,
                "acuracia": a,
                "sensibilidade": s,
                "especificidade": e,
                "pesos": ppn.getWeights(),
                "matriz_confusao": ppn.getMatrizConfusao()
            }
        )
        rounds += 1
        ppn = None
    printProgressBar(100, 'Concluído !!!')
    estatisticas = pd.DataFrame(rodada)
    calculos.append(
        {
            "Perceptron simples":
                {
                    "acuracia": [round(num.mean(estatisticas.iloc[:]["acuracia"]), 6), round(num.median(estatisticas.iloc[:]["acuracia"]), 6), round(num.min(estatisticas.iloc[:]["acuracia"]), 6), round(num.max(estatisticas.iloc[:]["acuracia"]), 6), round(num.std(estatisticas.iloc[:]["acuracia"]), 6)],
                    "especificidade": [round(num.mean(estatisticas.iloc[:]["especificidade"]), 6), round(num.median(estatisticas.iloc[:]["especificidade"]), 6), round(num.min(estatisticas.iloc[:]["especificidade"]), 6), round(num.max(estatisticas.iloc[:]["especificidade"]), 6), round(num.std(estatisticas.iloc[:]["especificidade"]), 6)],
                    "sensibilidade": [round(num.mean(estatisticas.iloc[:]["sensibilidade"]), 6), round(num.median(estatisticas.iloc[:]["sensibilidade"]), 6), round(num.min(estatisticas.iloc[:]["sensibilidade"]), 6), round(num.max(estatisticas.iloc[:]["sensibilidade"]), 6), round(num.std(estatisticas.iloc[:]["sensibilidade"]), 6)]
                }
        }
    )
    resultados = pd.DataFrame(calculos)
    print()
    print()
    print(resultados.keys().values[0])
    print(pd.DataFrame(data=resultados.iloc[:]['Perceptron simples'][0], index=["media", "mediana", "minimo", "maximo", "d.padrao"]).T)

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        espiral = load_full_data(file_name="spiral.csv", columns=["x1", "x2", "y",], separator=",", ignore_header=False)
        aerogerador = load_full_data("aerogerador.dat", ["Vel", "Pot"], "\t", False)
        red_wine = load_full_data("winequality-red.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        white_wine = load_full_data("winequality-white.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        os.system("cls" if (platform.system() == "Windows") else "clear")  # 'Darwin' <- macOS, 'Linux', 'Windows'
        # print(platform.system())
        run(inputData=espiral)
        # run(inputData=aerogerador)
        # run(inputData=red_wine)
        # run(inputData=white_wine)
finally:
    close()