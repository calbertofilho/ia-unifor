import numpy as num
from utils.manipulation import sign
num.seterr(divide='ignore', invalid='ignore')

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

class Perceptron:
    def __init__(self, tx_apr: float, n_epochs: int) -> None:
        self.eta = tx_apr
        self.epocas = n_epochs
        self.matriz_confusao = num.zeros((2, 2), dtype=int)

    def EQM(self, y_real, y_pred) -> float:
        return num.square(num.subtract(y_real, y_pred)).mean()

    def treinamento(self, X: num.ndarray[float], y: num.ndarray[float]) -> num.ndarray[float]:
        N, p = X.shape
        X = X.T
        self.X_trn = num.concatenate((-num.ones((1, N)), X))
        self.y_trn = y
        erro = True
        epoca = 0
        # print("\nN =", N)
        # print("p =", p)
        # print("X =", X)
        # print("X_t\n", self.X_trn)
        # print("d\n", self.y_trn)
        W = num.random.rand(p+1, 1)
        while erro and (epoca < self.epocas):
            erro = False
            for i in range(N):
                x_t = self.X_trn[:, i].reshape((p+1, 1))
                u_t = (W.T @ x_t)
                y_t = sign(u_t)
                d_t = self.y_trn[i]
                e_t = int(d_t - y_t)
                W = W + ((e_t * x_t * self.eta) / 2)
                if y_t != d_t:
                    erro = True
                # print("k =", i)
                # print("xᵏ_t\n", x_t)
                # print("uᵏ_t =", u_t)
                # print("yᵏ_t =", y_t)
                # print("dᵏ_t =", d_t)
                # print("eᵏ_t =", e_t)
                # print("Wᵏ\n", W)
            epoca += 1
        return W

    def predicao(self, W: num.ndarray[float], X: num.ndarray[float], y: num.ndarray[float]) -> tuple[float, float, float, float]:
        N, p = X.shape
        X = X.T
        regressao = (len(num.unique(y)) != 2)
        eqm = 0
        self.X_tst = num.concatenate((-num.ones((1, N)), X))
        self.y_tst = y
        for i in range(N):
            x_t = self.X_tst[:, i].reshape(p+1, 1)
            u_t = (W.T @ x_t)
            y_t = sign(u_t)
            d_t = self.y_tst
            if regressao:
                 eqm += self.EQM(d_t, u_t)
            else:
                y_real = int(d_t[i][0])
                y_predito = y_t
                self.matriz_confusao[0 if (y_predito == -1) else 1, 0 if (y_real == -1) else 1] += 1
        if regressao:
            acuracia = 0
            sensibilidade = 0
            especificidade = 0
        else:
            VN: int = self.matriz_confusao[0, 0]
            VP: int = self.matriz_confusao[1, 1]
            FN: int = self.matriz_confusao[0, 1]
            FP: int = self.matriz_confusao[1, 0]
            acuracia = 0 if num.isnan((VP + VN) / (VP + VN + FP + FN)) else ((VP + VN) / (VP + VN + FP + FN))
            sensibilidade = 0 if num.isnan(VP / (VP + FN)) else (VP / (VP + FN))
            especificidade = 0 if num.isnan(VN / (VN + FP)) else (VN / (VN + FP))
            eqm = 0
        return acuracia, sensibilidade, especificidade, (eqm / (2*N))

    def getMatrizConfusao(self) -> num.ndarray[int]:
        return self.matriz_confusao
