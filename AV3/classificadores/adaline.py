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

    def ativacao(self, valor_entrada):
        # Função de ativação
        return 

    def treinamento(self, X, y) -> None:
        # Funcao de treinamento
        qtde_amostras, qtde_caracteristicas = X.shape
        self.pesos = np.random.uniform(size = qtde_caracteristicas+1, low = -1, high = 1)
        ...

    def _atualiza_pesos(self, amostra, y_atl, y_pred) -> None:
        # Funcao que atualiza os pesos
        ...

    def predicao(self, amostras_teste):
        # Funcao de teste
        y_predito = 0
        return y_predito

    def getPesos(self) -> np.ndarray[float]:
        return self.pesos

    def gerarMatrizConfusao(self, y_real: np.ndarray[int], y_predito: np.ndarray[int]) -> pd.DataFrame:
        df = pd.DataFrame({
            "y_teste": y_real,
            "y_predito": y_predito
        })
        return pd.crosstab(df["y_teste"], df["y_predito"], rownames=["Real"], colnames=["Previsto"])

    def calcularEQM(self, y_real: np.ndarray[int], y_predito: np.ndarray[int]) -> float:
        return
