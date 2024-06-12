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
# xˈ = [-1.00000  -8.22173  -0.95343]; com d  =  1.00000
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
        #print("N =", N)
        #print("p =", p)
        #print("X =", X)
        #print("y =", y)
        X_bias = num.concatenate((-num.ones((1, N)), X.T))
        print("X_t\n", X_bias)
        d = y.T
        print("d =", d)
        #print("X[0]_t =", X_t[0])
        #print("X[1]_t =", X_t[1])
        #print("X[2]_t =", X_t[2])
        #print("Xˈ_t =", X_t[:,0], "com dˈ_t =", d_t[:, 0])
        #print("X²_t =", X_t[:,1], "com d²_t =", d_t[:, 1])
        #print("X³_t =", X_t[:,2], "com d³_t =", d_t[:, 2])
        W = num.random.rand(N, 1)
        print("W_t =", W.T)
        erro = True
        epoca = 0
        while erro and (epoca < self.epocas):
            erro = False
            for i in range(N):
                x_t = X_bias[i]
                print("x_t =", x_t)
                u_t = W.T @ x_t
                print("u_t =", u_t)
                y_t = 1 if (u_t >= 0) else -1
                d_t = d[:, i]
                print("y_t == dˈ_t")
                print(y_t, "==", int(d_t))
                print(y_t == d_t)
                W += self.eta * (d_t - y_t) * x_t
                print(W)
                if y_t != d_t:
                    erro = True
            epoca += 1
            print("epoca =", epoca)
        return W
