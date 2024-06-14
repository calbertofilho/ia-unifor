import numpy as np
import pandas as pd

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

# EQM = 1/(2*N) * (y_real - y_predito)²

class Perceptron(object):

    def __init__(self, tx_aprendizado = 0.001, n_iteracoes = 100):
        # Construtor da classe
        self.eta = tx_aprendizado
        self.epocas = n_iteracoes
        self.pesos = None
        self.bias = None

    def ativacao(self, valor_entrada):
        # Função de ativação
        return np.where(valor_entrada >= 0, 1, -1)

    def treinamento(self, X, y):
        # Funcao de treinamento
        erro = True
        epoca = 0
        qtde_amostras, qtde_caracteristicas = X.shape
        self.pesos = np.random.uniform(size = qtde_caracteristicas, low = -0.5, high = 0.5)
        # print("pesos =", self.pesos)
        self.bias = -1
        while erro and (epoca < self.epocas):
            erro = False
            for indice, caracteristicas in enumerate(X):
                resultado = np.dot(caracteristicas, self.pesos) + self.bias
                y_predito = self.ativacao(resultado)
                self._atualiza_pesos(caracteristicas, y[indice], y_predito)
                # print("indice =", indice)
                # print("u_t =", resultado)
                # print("y_t =", y_predito)
                if y_predito != y[indice]:
                    erro = True
            epoca += 1

    def _atualiza_pesos(self, amostra, y_atl, y_pred):
        # Funcao que atualiza os pesos
        erro = y_atl - y_pred
        correcao = self.eta * erro
        self.pesos += correcao * amostra
        #self.bias += correcao
        # print("e_t =", erro)
        # print("correcao =", correcao)
        # print("pesos =", self.pesos)
        # print("bias =", self.bias)

    def predicao(self, amostras_teste):
        # Funcao de teste
        resultado = np.dot(amostras_teste, self.pesos) + self.bias
        y_predito = self.ativacao(resultado)
        return y_predito

    def getPesos(self) -> np.ndarray[float]:
        return self.pesos

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
