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

    @abstractmethod
    def getPesos(self) -> np.ndarray[float]:
        ...

    @abstractmethod
    def gerarMatrizConfusao(self, y_real: np.ndarray[int], y_predito: np.ndarray[int]) -> pd.DataFrame:
        ...

    @abstractmethod
    def calcularEQM(self, y_real: np.ndarray[int], y_predito: np.ndarray[int]) -> float:
        ...

    @abstractmethod
    def getCustos(self):
        ...
