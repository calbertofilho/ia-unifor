import numpy as np
import pandas as pd
from classificadores.classificador import Classificador

class Adaline(Classificador):
    def __init__(self, tx_aprendizado = 0.0001, n_iteracoes = 100, precisao = 1e-10):
        # Construtor da classe
        self.eta = tx_aprendizado
        self.epocas = n_iteracoes
        self.e = precisao

    # def eqm(self, X, y) -> float:
    #     eqm = 0
    #     for _, caracteristicas in enumerate(X):
    #         similaridade = np.dot(self.pesos.T, caracteristicas)                # u(t)
    #         eqm += np.square(y - similaridade).mean()
    #     return eqm / (2 * len(y))

    # def treinamento(self, X, y) -> None:
    #     # Funcao de treinamento
    #     qtde_amostras, qtde_caracteristicas = X.shape
    #     X1 = np.append(np.ones(qtde_amostras).reshape(qtde_amostras, 1), X, axis = 1)
    #     self.pesos = np.random.uniform(size = (qtde_caracteristicas + 1, 1), low = -1, high = 1)
    #     final = self.epocas
    #     eqm_ = self.eqm(X1, y)
    #     self.custos = []
    #     for epoca in range(self.epocas):
    #         for indice, caracteristicas in enumerate(X1):
    #             similaridade = np.dot(self.pesos.T, caracteristicas)            # u(t)
    #             self.pesos = self.pesos + (self.eta * (y[indice] - similaridade) * caracteristicas)
    #         novo_eqm = self.eqm(X1, y)
    #         if ((novo_eqm - eqm_) >= self.e):
    #             final = epoca+1
    #             break
    #         self.custos.append(eqm_)
    #     return final

    def ativacao(self, amostras):
        # Função de ativação
        return np.dot(amostras, self.pesos[1:]) + self.pesos[0]

    def treinamento(self, X, y) -> None:
        # Funcao de treinamento
        qtde_amostras, qtde_caracteristicas = X.shape
        self.pesos = np.random.uniform(low = -1, high = 1, size = (qtde_caracteristicas + 1)).reshape((qtde_caracteristicas + 1), 1)
        final = self.epocas
        custo = 0
        self.custos = []
        for _ in range(self.epocas):
            resultado = self.ativacao(X).reshape(qtde_amostras, 1)
            erro = (y - resultado)
            self.pesos[0] += self.eta * erro.sum()
            self.pesos[1:] += self.eta * X.T.dot(erro)
            custo = np.square(erro).sum() / 2.
            self.custos.append(custo)
        return final

    def predicao(self, amostras_teste):
        # Funcao de teste
        return np.where(self.ativacao(amostras_teste) >= 0.0, 1, -1)

    def getPrecisao(self):
        return self.e
