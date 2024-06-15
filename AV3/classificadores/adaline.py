import numpy as np
import pandas as pd

# https://medium.com/ensina-ai/rede-neural-perceptron-adaline-8f69dc419d4e
# https://sebastianraschka.com/faq/docs/diff-perceptron-adaline-neuralnet.html

class Adaline(object):
    def __init__(self, tx_aprendizado = 0.001, n_iteracoes = 100):
        # Construtor da classe
        self.eta = tx_aprendizado
        self.epocas = n_iteracoes
        self.pesos = None
        self.bias = None

    def ativacao(self, amostras):
        # Função de ativação
        return np.dot(amostras, self.pesos[1:]) + self.pesos[0]

    def treinamento(self, X, y) -> None:
        # Funcao de treinamento
        qtde_amostras, qtde_caracteristicas = X.shape
        self.pesos = np.random.uniform(low = -1, high = 1, size = (qtde_caracteristicas + 1)).reshape((qtde_caracteristicas + 1), 1)
        self.custos = []
        custo = 0
        for _ in range(self.epocas):
            resultado = self.ativacao(X).reshape(qtde_amostras, 1)
            erro = (y - resultado)
            self.pesos[0] += self.eta * erro.sum()
            self.pesos[1:] += self.eta * X.T.dot(erro)
            custo = np.square(erro).sum() / 2.
            self.custos.append(custo)

    def _atualiza_pesos(self, amostra, y_atl, y_pred) -> None:
        # Funcao que atualiza os pesos
        ...

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
