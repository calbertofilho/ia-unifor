import numpy as num
from utils.manipulation import sign
num.seterr(divide='ignore', invalid='ignore')

class Perceptron:
    def __init__(self, tx_apr: float, n_epochs: int) -> None:
        self.eta = tx_apr
        self.epocas = n_epochs
        self.matriz_confusao = num.zeros((2, 2), dtype=int)

    def EQM(y_real, y_pred) -> float:
        return num.square(num.subtract(y_real, y_pred)).mean()

    def treinar(self, X: num.ndarray[float], y: num.ndarray[float]) -> num.ndarray[float]:
        N, p = X.shape
        X = X.T
        self.X_trn = num.concatenate((-num.ones((1, N)), X))
        self.y_trn = y
        erro = True
        epoca = 0
        W = num.random.rand(p+1, 1)
        while erro and (epoca < self.epocas):
            erro = False
            for i in range(N):
                x_t = self.X_trn[:, i].reshape((p+1, 1))
                u_t = (W.T @ x_t)[0, 0]
                y_t = sign(u_t)
                d_t = self.y_trn[i, 0]
                e_t = int(d_t - y_t)
                W = W + ((e_t * x_t * self.eta) / 2)
                if y_t != d_t:
                    erro = True
            epoca += 1
        return W

    def testar(self, W: num.ndarray[float], X: num.ndarray[float], y: num.ndarray[float]) -> tuple[float, float, float, float]:
        N, p = X.shape
        X = X.T
        eqm = 0
        self.X_tst = num.concatenate((-num.ones((1, N)), X))
        self.y_tst = y
        for i in range(N):
            x_t = self.X_tst[:, i].reshape(p+1, 1)
            u_t = (W.T @ x_t)
            y_t = sign(u_t[0, 0])
            d_t = self.y_tst
            eqm += num.square(num.subtract(d_t, u_t))
            y_real = int(d_t[i][0])
            y_predito = y_t
            self.matriz_confusao[0 if (y_predito == -1) else 1, 0 if (y_real == -1) else 1] += 1
        VN: int = self.matriz_confusao[0, 0]
        VP: int = self.matriz_confusao[1, 1]
        FN: int = self.matriz_confusao[0, 1]
        FP: int = self.matriz_confusao[1, 0]
        acuracia = 0 if num.isnan((VP + VN) / (VP + VN + FP + FN)) else ((VP + VN) / (VP + VN + FP + FN))
        sensibilidade = 0 if num.isnan(VP / (VP + FN)) else (VP / (VP + FN))
        especificidade = 0 if num.isnan(VN / (VN + FP)) else (VN / (VN + FP))
        return acuracia, sensibilidade, especificidade, (eqm / N)

    def getMatrizConfusao(self) -> num.ndarray[int]:
        return self.matriz_confusao
