import numpy as np
import pandas as pd
from classificadores.classificador import Classificador

class Adaline(Classificador):
    def __init__(self, tx_aprendizado = 0.0001, n_iteracoes = 100):
        # Construtor da classe
        self.eta = tx_aprendizado
        self.epocas = n_iteracoes

    def ativacao(self, amostras):
        # Função de ativação
        return np.dot(amostras, self.pesos[1:]) + self.pesos[0]

    def treinamento(self, X, y) -> None:
        # Funcao de treinamento
        qtde_amostras, qtde_caracteristicas = X.shape
        self.pesos = np.random.uniform(low = -1, high = 1, size = (qtde_caracteristicas + 1)).reshape((qtde_caracteristicas + 1), 1)
        custo = 0
        self.custos = []
        for _ in range(self.epocas):
            resultado = self.ativacao(X).reshape(qtde_amostras, 1)
            erro = (y - resultado)
            self.pesos[0] += self.eta * erro.sum()
            self.pesos[1:] += self.eta * X.T.dot(erro)
            custo = np.square(erro).sum() / 2.
            self.custos.append(custo)

    def predicao(self, amostras_teste):
        # Funcao de teste
        return np.where(self.ativacao(amostras_teste) >= 0.0, 1, -1)

    def getPesos(self) -> np.ndarray[float]:
        return self.pesos

    def gerarMatrizConfusao(self, y_real: np.ndarray[int], y_predito: np.ndarray[int]) -> pd.DataFrame:
        df = pd.DataFrame({
            "y_teste": y_real,
            "y_predito": y_predito
        })
        return pd.crosstab(df["y_teste"], df["y_predito"], rownames=["Real"], colnames=["Previsto"])

    def calcularEQM(self, y_real: np.ndarray[int], y_predito: np.ndarray[int]) -> float:
        return np.square(np.subtract(y_real, y_predito)).mean() / (2 * len(y_real))

    def getCustos(self):
        return self.custos
