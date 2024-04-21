import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Rastrigin
#                p
# f(x) = A · p + ∑ (x[i]² − A · cos(2 · π · x[i]))
#               i=1

limites = [(-10, 10)] * 2
def funcao(x1, x2):
#     def rastriginFunc(x):
#         rastrigin = A * len(x)
#         for i in range(len(x)):
#             rastrigin += math.pow(x[i], 2) - (A * np.cos(2 * np.pi * x[i]))
#         return rastrigin
#     return rastriginFunc(x) + 1
    return ((np.power(x1, 2) - 10 * np.cos(2 * np.pi * x1) + 10) + (np.power(x2, 2) - 10 * np.cos(2 * np.pi * x2) + 10))

# Geração do grid e gráfico da função
x = np.linspace(start=[limites[0][0], limites[1][0]], stop=[limites[0][1], limites[1][1]], num=1000, axis=1)
X1, X2 = np.meshgrid(x[0], x[1])
Y = funcao(X1, X2)

# Plotagem do desenho do gráfico e do ponto inicial
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
# plot.colorbar(surface)

# Etiquetas dos eixos
ax.set_title("rastrigin")
ax.set_xlabel('valores x')
ax.set_ylabel('valores y')
ax.set_zlabel('valores z')
plt.tight_layout()  # Melhora o ajuste para a imagem a ser plotada

ax.view_init(elev=30, azim=-65, roll=0.)
plt.show()



# A = 10
# N = 30
# p = 20
# nBits = 5
# xLow = -10.0
# xHigh = 10.0
# recombinationProb = 0.85
# mutationProb = 0.01
# maxGeneration = 100
# population = None
# selection = None
# recombination = None
# fitness = np.zeros(N)
# fitnessSum = 0
# best = []
# average = []

# def fitnessFunc(x:np.ndarray[int]) -> float:
#     def rastriginFunc(x):
#         rastrigin = A * len(x)
#         for i in range(len(x)):
#             rastrigin += math.pow(x[i], 2) - (A * np.cos(2 * np.pi * x[i]))
#         return (A * p) + rastrigin
#     return rastriginFunc(x) + 1

# def population() -> np.ndarray[int]:
#     return np.random.randint(low=0, high=2, size=(N, p * nBits))

# def decode(x:np.ndarray[int]) -> int:
#     dec = 0
#     for i in range(len(x)):
#         dec += x[len(x) - 1 - i] * math.pow(2, i)
#     return xLow + ((xHigh - xLow) / (math.pow(2, nBits) - 1)) * dec

# def calculateFitness() -> None:
#     for i in range(N):
#         x, y = decode(population[i, 0:nBits]), decode(population[i, nBits:])
#         fitness[i] = fitnessFunc(x, y)
#     fitnessSum = np.sum(fitness)
#     best.append(np.max(fitness))
#     average.append(np.mean(fitness))

# def roulette() -> np.ndarray[int]:
#     i = 0
#     amount = fitness[i] / fitnessSum
#     r = np.random.uniform()
#     while amount < r:
#         i += 1
#         amount += fitness[i] / fitnessSum
#     return population[i, :]

# def selection() -> np.ndarray[int]:
#     sel = np.empty((0, nBits * p))
#     for _ in range(N):
#         s = roulette()
#         sel = np.concatenate((sel, s.reshape(1, nBits * p)))
#     return sel

# def recombination() -> np.ndarray[int]:
#     R = np.empty((0, nBits * p))
#     for i in range(0, N, 2):
#         x1 = selection[i, :]
#         x2 = selection[i + 1, :]
#         x1_ = np.copy(x1)
#         x2_ = np.copy(x2)
#         if(np.random.uniform() <= recombinationProb):
#             m = np.zeros(p * nBits)
#             xi = np.random.randint(0, p * nBits - 1)
#             m[xi + 1:] = 1
#             x1_[m[:]==1] = x2[m[:]==1]
#             x2_[m[:]==1] = x1[m[:]==1]
#         R = np.concatenate((R, x1_.reshape(1, p * nBits), x2_.reshape(1, p * nBits),))
#     return R

# def toggle(b:bool) -> bool:
#     return 1 if b == 0 else 0

# def mutation() -> None:
#     for i in range(N):
#         for j in range(nBits * p):
#             if np.random.uniform() <= mutationProb:
#                 population[i, j] = toggle(population[i, j])

# def generation() -> None:
#     population = population()
#     for _ in range(maxGeneration):
#         calculateFitness()
#         selection = selection()
#         population = recombination()
#         mutation()

def run() -> None:
    ...

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        run()
finally:
    close()
