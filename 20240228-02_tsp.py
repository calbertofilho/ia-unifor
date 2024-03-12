# Traveling Salesman Problem 2

import numpy as np
import matplotlib.pyplot as plt

def perturb(x, e):
    return x

def f(x):
    return np.exp(-x ** 2)

x_otimo = np.random.uniform(low=-2, high=2)
f_otimo = f(x_otimo)

e = 1
max_it = 1000
max_vizinhos = 10
i = 0
melhoria = True

while i<max_it and melhoria:
    melhoria = False
    for j in range(max_vizinhos):
        x_candidato = perturb(x_otimo, e)
        f_candidato = f(x_candidato)

plt.scatter(x_otimo, f_otimo, color='red', s=90, marker='x')
x_axis = np.linspace(-2, 2, 1000)
plt.plot(x_axis, f(x_axis))
plt.grid()
#plt.show()
plt.savefig('20240228-02_tsp.png')