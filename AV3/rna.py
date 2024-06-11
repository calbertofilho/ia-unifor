import sys
import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot
from algoritmos.mlp import MLP
from algoritmos.adaline import Adaline
from algoritmos.perceptron import Perceptron
from algoritmos.percept import Perceptron as Percept
from utils.manipulation import clearScreen, loadData, shuffleData, partitionData
from utils.progress import printProgressBar, printAnimatedBar

def new(inputData: pd.DataFrame) -> None:
    percept = Percept(n_features=2)
    data = shuffleData(inputData)
    X_treino, X_teste, y_treino, y_teste = partitionData(data, 0.8)
    percept.training(X_treino, y_treino)
    y_predict = percept.predict(X_teste)
    print("y_predito = ", y_predict)
    print("y_teste = ", y_teste)
    print(percept.showAccuracy(y_teste, y_predict))

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
        pesos = ppn.treinar(X=X_trn, y=y_trn)
        a, s, e, q = ppn.testar(pesos, X_tst, y_tst)
        rodada.append(
            {
                "rodada": rounds+1,
                "acuracia": a,
                "sensibilidade": s,
                "especificidade": e,
                "eqm": q,
                "pesos": pesos,
                "matriz_confusao": ppn.getMatrizConfusao()
            }
        )
        rounds += 1
        ppn = data = X_trn = X_tst = y_trn = y_tst = pesos = None
        a = s = e = q = 0
    printProgressBar(100, 'Concluído !!!')
    estatisticas = pd.DataFrame(rodada)
    calculos.append(
        {
            "Perceptron simples":
                {
                    "eqm": [round(num.mean(estatisticas.iloc[:]["eqm"]), 6), round(num.median(estatisticas.iloc[:]["eqm"]), 6), round(num.min(estatisticas.iloc[:]["eqm"]), 6), round(num.max(estatisticas.iloc[:]["eqm"]), 6), round(num.std(estatisticas.iloc[:]["eqm"]), 6)],
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
        # run(inputData=espiral)
        # run(inputData=aerogerador)
        # run(inputData=red_wine)
        # run(inputData=white_wine)
        new(inputData=espiral)
finally:
    close()


    # EQM = 1/N (y_real - y_predito)²