import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot

class MLP:
    def __init__(self, data: pd.DataFrame) -> None:
        self.setDados(data)

    def setDados(self, data: pd.DataFrame) -> None:
        self.dados = data

    def getDados(self) -> pd.DataFrame:
        return self.dados

    def getShape(self) -> tuple[int, int]:
        return self.dados.shape