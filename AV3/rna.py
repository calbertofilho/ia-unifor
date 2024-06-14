import sys
import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot
from classificadores.mlp import MultilayerPerceptron as MLP
from classificadores.adaline import Adaline as Ada
from classificadores.perceptron import Perceptron
from utils.manipulation import clearScreen, loadData, shuffleData, partitionData
from utils.progress import printProgressBar, printAnimatedBar

def run(inputData: pd.DataFrame, algoritmo: object) -> None:
    rodada = 0                                                                  # Contador das rodadas
    rodadas = 100                                                               # Número máximo de rodadas
    dados_rodada = []                                                           # Coleta de dados de cada rodada
    calculos = []                                                               # Armazena os cálculos
    classificador = algoritmo                                                   # Algoritmo que vai executar a classificação
    while (rodada < rodadas):
        printProgressBar((rodada / rodadas) * 100, 'Calculando...')
        data = shuffleData(inputData)                                           # Embaralha os dados
        X_trn, X_tst, y_trn, y_tst = partitionData(data, 0.8)                   # Particiona os dados no percentual proposto
        classificador.treinamento(X_trn, y_trn)                                 # Treina o classificador com os dados separados para treinamento
        y = num.array(y_tst, dtype=int).flatten()                               # Organiza os rotulos da amostra de teste
        y_ = classificador.predicao(X_tst)                                      # Calcula a predição da amostra de teste
        eqm = classificador.calcularEQM(y, y_)                                  # Calcula o EQM
        rodada += 1
        if len(num.unique(y_tst)) == 2:                                         # Testa se é uma classificação (apenas dois rótulos)
            matriz = classificador.gerarMatrizConfusao(y, y_)                   # Gera a matriz de confusão
            # print(matriz)
            VN = int(matriz.loc[1].loc[1])                                      # Valores encontrados como VERDADEIROS NEGATIVOS
            VP = int(matriz.loc[-1].loc[-1])                                    # Valores encontrados como VERDADEIROS POSITIVOS
            FN = int(matriz.loc[-1].loc[1])                                     # Valores encontrados como FALSOS NEGATIVOS
            FP = int(matriz.loc[1].loc[-1])                                     # Valores encontrados como FALSOS POSITIVOS
            # print("VN =", VN)
            # print("VP =", VP)
            # print("FN =", FN)
            # print("FP =", FP)
            dados_rodada.append({
                "rodada": rodada,
                "acuracia": (VP + VN) / (VP + VN + FP + FN),
                "sensibilidade": VP / (VP + FN),
                "especificidade": VN / (VN + FP),
                "eqm": eqm,
                "pesos": classificador.getPesos(),
                "matriz_confusao": matriz
            })                                                                  # Armazena os dados da rodada
        else:
            dados_rodada.append({
                "rodada": rodada,
                "eqm": eqm,
                "pesos": classificador.getPesos()
            })                                                                  # Armazena os dados da rodada
    printProgressBar(100, 'Concluído !!!')
    dados = pd.DataFrame(dados_rodada)                                          # Organiza os dados para manipulação
    if "acuracia" in dados.columns:                                             # Se encontrar a coluna 'acuracia' no DataFrame, pega os dados referente a classificação
        calculos.append(
            {
                "Perceptron simples":
                    {
                        "eqm": [
                            "{:.4f}".format(num.mean(dados.iloc[:]["eqm"])),
                            "{:.4f}".format(num.median(dados.iloc[:]["eqm"])),
                            "{:.4f}".format(num.min(dados.iloc[:]["eqm"])),
                            "{:.4f}".format(num.max(dados.iloc[:]["eqm"])),
                            "{:.4f}".format(num.std(dados.iloc[:]["eqm"]))
                        ],
                        "acuracia": [
                            "{:.4f}".format(num.mean(dados.iloc[:]["acuracia"])),
                            "{:.4f}".format(num.median(dados.iloc[:]["acuracia"])),
                            "{:.4f}".format(num.min(dados.iloc[:]["acuracia"])),
                            "{:.4f}".format(num.max(dados.iloc[:]["acuracia"])),
                            "{:.4f}".format(num.std(dados.iloc[:]["acuracia"]))
                        ],
                        "especificidade": [
                            "{:.4f}".format(num.mean(dados.iloc[:]["especificidade"])),
                            "{:.4f}".format(num.median(dados.iloc[:]["especificidade"])),
                            "{:.4f}".format(num.min(dados.iloc[:]["especificidade"])),
                            "{:.4f}".format(num.max(dados.iloc[:]["especificidade"])),
                            "{:.4f}".format(num.std(dados.iloc[:]["especificidade"]))
                        ],
                        "sensibilidade": [
                            "{:.4f}".format(num.mean(dados.iloc[:]["sensibilidade"])),
                            "{:.4f}".format(num.median(dados.iloc[:]["sensibilidade"])),
                            "{:.4f}".format(num.min(dados.iloc[:]["sensibilidade"])),
                            "{:.4f}".format(num.max(dados.iloc[:]["sensibilidade"])),
                            "{:.4f}".format(num.std(dados.iloc[:]["sensibilidade"]))
                        ]
                    }
            }
        )
    else:                                                                       # Se não encontrar, pega os dados referente a regressão
        calculos.append(
            {
                "Perceptron simples":
                    {
                        "eqm": [
                            "{:.4f}".format(num.mean(dados.iloc[:]["eqm"])),
                            "{:.4f}".format(num.median(dados.iloc[:]["eqm"])),
                            "{:.4f}".format(num.min(dados.iloc[:]["eqm"])),
                            "{:.4f}".format(num.max(dados.iloc[:]["eqm"])),
                            "{:.4f}".format(num.std(dados.iloc[:]["eqm"]))
                        ]
                    }
            }
        )
    resultados = pd.DataFrame(calculos)
    print("\n\n", resultados.keys().values[0])
    print(pd.DataFrame(data=resultados.iloc[:]['Perceptron simples'][0], index=["media", "mediana", "minimo", "maximo", "d.padrao"]).T)
    # print("dados_rodada\n", dados_rodada)
    # print("dados\n", dados)
    # print("calculos\n", calculos)
    # print("resultados\n", resultados)

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        espiral = loadData(file_name="spiral.csv", columns=["x1", "x2", "y",], separator=",", ignore_header=False)
        aerogerador = loadData("aerogerador.dat", ["Vel", "Pot"], "\t", False)
        red_wine = loadData("winequality-red.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        white_wine = loadData("winequality-white.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        percecptron = Perceptron(tx_aprendizado=0.001, n_iteracoes=100)         # Inicia o classificador com a taxa de aprendizado e o número de épocas para iterações
        clearScreen()
        run(inputData=espiral, algoritmo=percecptron)
        # run(inputData=aerogerador, algoritmo=percecptron)
        # run(inputData=red_wine, algoritmo=percecptron)
        # run(inputData=white_wine, algoritmo=percecptron)
finally:
    close()
