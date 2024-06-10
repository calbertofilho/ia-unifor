import os
import sys
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
    print()
    print(num.mean(estatisticas.iloc[:]["acuracia"]))
    print(num.median(estatisticas.iloc[:]["acuracia"]))
    print(num.min(estatisticas.iloc[:]["acuracia"]))
    print(num.max(estatisticas.iloc[:]["acuracia"]))
    print(num.std(estatisticas.iloc[:]["acuracia"]))

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        espiral = load_full_data(file_name="spiral.csv", columns=["x1", "x2", "y",], separator=",", ignore_header=False)
        aerogerador = load_full_data("aerogerador.dat", ["Vel", "Pot"], "\t", False)
        red_wine = load_full_data("winequality-red.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        white_wine = load_full_data("winequality-white.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        os.system("cls" if (sys.platform in ("win32", "win64")) else "clear")  #'darwin' <- macOS, 'linux', ('win32', 'win64')
        print(sys.platform)
        run(inputData=espiral)
finally:
    close()