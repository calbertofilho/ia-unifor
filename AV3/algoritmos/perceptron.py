import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot

class Perceptron:
    def __init__(self, data: pd.DataFrame) -> None:
        self.setDados(data)

    def setDados(self, data: pd.DataFrame) -> None:
        self.dados = data

    def getDados(self) -> pd.DataFrame:
        return self.dados

    def getShape(self) -> tuple[int, int]:
        return self.dados.shape

    def shuffleDados(self) -> None:
        self.setDados(self.dados.sample(frac=1).reset_index(drop=True))

    def partitionDados(self, percentual: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        X = self.dados.iloc[:int((self.dados.tail(1).index.item()+1)*percentual), :len(self.dados.axes[1])-1].values
        y = self.dados.iloc[:int((self.dados.tail(1).index.item()+1)*percentual), len(self.dados.axes[1])-1:].values
        X_rest = self.dados.iloc[int((self.dados.tail(1).index.item()+1)*percentual):, :len(self.dados.axes[1])-1].values
        y_rest = self.dados.iloc[int((self.dados.tail(1).index.item()+1)*percentual):, len(self.dados.axes[1])-1:].values
        return X, X_rest, y, y_rest
