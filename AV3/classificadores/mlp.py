import numpy as np
import pandas as pd
from classificadores.classificador import Classificador

# para uma iteração completa (treinamento ou teste) faça
#   propagação(data)
#   volta(data)
#   ajusta_pesos()

class MultilayerPerceptron(Classificador):
    def __init__(self, tx_aprendizado = 0.0001, n_iteracoes = 100, n_camadas = 3):
        # Construtor da classe
        self.eta = tx_aprendizado
        self.epocas = n_iteracoes
        self.camadas_escondidas = n_camadas

    def ativacao(self, amostras):
        # Função de ativação
        return 

    def _forward(self, X):
        # Funcao que propaga os dados na rede
        passo = []
        for i in range(self.camadas_escondidas):
            passo[i] = self._forward(X)
        return passo[self.camadas_escondidas]

    def _backward(self, y, step):
        # Funcao que retorna na rede ajustando os pesos
        for indice, rotulos in enumerate(y):
            self.pesos = np.dot(..., y)
            resultado = np.dot(rotulos, self.weights.T)
        ...
        return resultado

    def treinamento(self, X, y) -> None:
        # Funcao de treinamento
        for _ in range(self.epocas):
            for indice, caracteristicas in enumerate(X):
                output = self._propagation(caracteristicas, y[indice])
        ...

    def predicao(self, amostras_teste):
        # Funcao de teste
        return 

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
