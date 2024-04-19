# Traveling Salesman Problem '2D'

import numpy as np
import matplotlib.pyplot as plt

def perturb(x,e):
    return np.random.uniform(low=x-e, high=x+e)

def f(x):
    return np.exp(-x ** 2)

x_axis = np.linspace(-2, 2, 1000)
x_otimo = np.random.uniform(low=-2, high=2)
f_otimo = f(x_otimo)

delay=.5
e = .1
max_it = 1000
max_vizinhos = 20
i = 0
melhoria = True

plt.plot(x_axis, f(x_axis))
plt.grid()
plt.suptitle('Realizando busca...', fontsize = 12)
plt.scatter(x_otimo, f_otimo, color='green', s=120, marker='x', linewidth=1)

while i<max_it and melhoria:
    melhoria = False
    for j in range(max_vizinhos):
        x_candidato = perturb(x_otimo, e)
        f_candidato = f(x_candidato)
        if(f_candidato>f_otimo):
            melhoria = True
            x_otimo = x_candidato
            f_otimo = f_candidato
            plt.pause(delay)
            plt.title("x = "+"{:.4f}".format(x_otimo)+"\nf(x) = "+"{:.4f}".format(f_otimo), fontsize = 8)
            plt.scatter(x_otimo, f_otimo, color='black', s=20, marker='x', linewidth=1)
            break
    i+=1

plt.pause(delay)
plt.scatter(x_otimo, f_otimo, color='red', s=120, marker='x', linewidth=1)

plt.suptitle('Solução ótima encontrada!', fontsize = 12)
plt.savefig('20240228-02_tsp.png')
plt.show()