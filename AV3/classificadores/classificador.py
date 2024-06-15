import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class Classificador(ABC):

    @abstractmethod
    def ativacao(self, valor_entrada):
        ...

    @abstractmethod
    def treinamento(self, X, y) -> int:
        ...

    @abstractmethod
    def predicao(self, amostras_teste) -> int:
        ...

    def gerarMatrizConfusao(self, y_real: np.ndarray[int], y_predito: np.ndarray[int]) -> pd.DataFrame:
        #  Previsto  -->  -1   1    |    VP    [0, 0]  Verdadeiro positivo
        #  Real                     |    VN    [1, 1]  Verdadeiro negativo
        #     |     -1    VP  FN    |    FN    [0, 1]  Falso negativo
        #     V      1    FP  VN    |    FP    [1, 0]  Falso positivo
        # res = np.zeros((2, 2), dtype=int)
        # for a, p in zip(y_real, y_predito):
        #     res[a][p] += 1
        # return res
        df = pd.DataFrame({
            "y_teste": y_real,
            "y_predito": y_predito
        })
        return pd.crosstab(df["y_teste"], df["y_predito"], rownames=["Real"], colnames=["Previsto"])

    def calcularEQM(self, y_real: np.ndarray[int], y_predito: np.ndarray[int]) -> float:
        return np.square(np.subtract(y_real, y_predito)).mean() / (2 * len(y_real))

    def getCustos(self) -> list[float]:
        return self.custos

    def getPesos(self) -> np.ndarray[float]:
        return self.pesos
