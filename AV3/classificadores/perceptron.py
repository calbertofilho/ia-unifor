import numpy as np
import pandas as pd
from classificadores.classificador import Classificador

# Dados
#       -8.22173   -0.95343    1.00000
#        1.05714    4.79365    1.00000
#       -7.40609   11.14205   -1.00000
#
# Separa X de y
# Transpões X e concatena o BIAS na primeira linha
#
# X.T         x          x²         x³
#   X0    -1.00000   -1.00000   -1.00000
#   X1    -8.22173    1.05714   -7.40609
#   X2    -0.95343    4.79365   11.14205
# y.T         d          d²         d³
#   d_t    1.00000    1.00000   -1.00000
#
# xˈ = [-1.00000  -8.22173  -0.95343]; com d  =  1.00000
# x² = [-1.00000   1.05714   4.79365]; com d² =  1.00000
# x³ = [-1.00000  -7.40609  11.14205]; com d³ = -1.00000
#
# w = rand(0, 1)
# u = W.T @ Xᵏ
# y = sinal(u)
# se y != dᵏ
#   W += ƞ(dᵏ - y)xᵏ
#
# EQM = 1/(2*N) * (y_real - y_predito)²

class Perceptron(Classificador):
    def __init__(self, tx_aprendizado = 0.5, n_iteracoes = 100):
        # Construtor da classe
        self.eta = tx_aprendizado
        self.epocas = n_iteracoes

    def ativacao(self, valor_entrada):
        # Função de ativação
        return np.where(valor_entrada >= 0, 1, -1)

    def treinamento(self, X, y) -> int:
        # Funcao de treinamento
        qtde_amostras, qtde_caracteristicas = X.shape
        X1 = np.append(np.ones(qtde_amostras).reshape(qtde_amostras, 1), X, axis = 1)
        self.pesos = np.random.uniform(size = qtde_caracteristicas + 1, low = -1, high = 1)
        final = self.epocas
        custo = 0
        concluido = True
        self.custos = []
        for epoca in range(self.epocas):
            concluido = True
            for indice, caracteristicas in enumerate(X1):
                similaridade = np.dot(caracteristicas, self.pesos.T)            # u(t)
                resultado = self.ativacao(similaridade)                         # d(t)
                self.pesos += self.eta * caracteristicas * np.subtract(y[indice], resultado)
                custo += similaridade
                if resultado != y[indice]:                                      # d(t) != y(t)
                    concluido = False
            self.custos.append(custo)
            if concluido:
                final = epoca+1
                break
        return final

    def predicao(self, amostras_teste) -> int:
        # Funcao de teste
        qtde_amostras, _ = amostras_teste.shape
        X = np.append(np.ones(qtde_amostras).reshape(qtde_amostras, 1), amostras_teste, axis = 1)
        resultado = np.dot(X, self.pesos.T)
        return self.ativacao(resultado)
