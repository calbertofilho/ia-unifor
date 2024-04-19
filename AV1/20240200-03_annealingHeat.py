import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot

def f(x, y):
    return x ** 2 * num.sin(4 * num.pi * x) - y * num.sin(4 * num.pi * y + num.pi) + 1
                                            
x_axis = num.linspace(-1, 2, 1000)

X,Y = num.meshgrid(x_axis, x_axis)

Z  = f(X, Y)

x_l = num.array([-1, -1]) 
x_u = num.array([2, 2]) 

fig = plot.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(X, Y, Z, cmap='gray', alpha=.1, rstride=30, cstride=30, edgecolor='k')

x_opt = num.random.uniform(-1, 2, size=(2, ))
f_opt = f(x_opt[0], x_opt[1])

ax.scatter(x_opt[0], x_opt[1], f_opt, color='red', s=120, marker="x", linewidth=1)
T = 100
sigma = .5
heuristicas = []

for i in range(1000):
    n = num.random.normal(0, scale=sigma, size=(2,))
    x_cand = x_opt + n

    for j in range(x_cand.shape[0]):
        if(x_cand[j] < x_l[j]):
            x_cand[j] = x_l[j]
        if(x_cand[j] > x_u[j]):
            x_cand[j] = x_u[j]

    f_cand = f(x_cand[0], x_cand[1])
    P_ij = num.exp(-(f_cand - f_opt) / T)
    heuristicas.append(f_opt)
    if ((f_cand < f_opt) or (P_ij >= num.random.uniform(0,1))):
        x_opt = x_cand
        f_opt = f_cand
    
        ax.scatter(x_opt[0], x_opt[1], f_opt, color='blue', s=20, marker="o", linewidth=1)

    T = T * .89

ax.scatter(x_opt[0], x_opt[1], f_opt, color='green', s=150, marker="*", linewidth=1)
plot.show()

plot.plot(heuristicas)
plot.show()