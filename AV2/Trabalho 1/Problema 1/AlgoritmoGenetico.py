import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#                p
# f(x) = A · p + ∑ (x[i]² − A · cos(2 · π · x[i]))
#               i=1

A = 10
N = 30
p = 20
nBits = 5
xLow = -10.0
xHigh = 10.0
recombinationProb = 0.85
mutationProb = 0.01

def fitnessFunc(x):
    def rastriginFunc(x):
        result = A * len(x)
        for i in range(len(x)):
            result += math.pow(x[i], 2) - (A * np.cos(2 * np.pi * x[i]))
        return (A * p) + result
    return rastriginFunc(x) + 1

def population():
    return np.random.randint(low=0, high=2, size=(N, p * nBits))

def roulette():
    return

def selection():
    return

def calculateFitness():
    return

def recombination():
    return

def toggle(b):
    return 1 if b == 0 else 0

def mutation():
    return

def generation():
    return
