import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot

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
# x  = [-1.00000  -8.22173  -0.95343]; com d  =  1.00000
# x² = [-1.00000   1.05714   4.79365]; com d² =  1.00000
# x³ = [-1.00000  -7.40609  11.14205]; com d³ = -1.00000
#
# w = rand(0, 1)
# u = W.T @ Xᵏ
# y = sinal(u)
# se y != dᵏ
#   W += ƞ(dᵏ - y)xᵏ

class Perceptron(object):

    def __init__(self, tx_apr=0.001, n_iter=1000):
        self.eta = tx_apr
        self.epocas = n_iter

    def treinamento(self, X: num.ndarray[float], y: num.ndarray[float]) -> num.ndarray[float]:
        N, p = X.shape
        print("N =", N)
        print("p =", p)
        print("X =", X)
        print("y =", y)
        X_t = num.concatenate((-num.ones((1, N)), X.T))
        print("X_t =", X_t)
        print(X_t.shape)
        d_t = y.T
        print("d_t =", d_t)
        print(d_t.shape)
        W = num.random.rand(N, 1)
        print("W_t =", W.T)
        print(W.T.shape)
        u_t = W.T @ X_t[0,:]
        print("u_t = ", u_t)
        y_t = 1 if (u_t >= 0) else -1
        print("y_t =", y_t)
        W += self.eta * (d_t - y_t) * x_t
        return W
