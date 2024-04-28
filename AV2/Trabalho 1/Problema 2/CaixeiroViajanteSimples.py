import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

# DEFINIÇÕES
# 
# Gene        Gene        Ponto no espaço representado por coordenadas (x, y, z)
N = 100
# Individual  Indivíduo   Rota única que satisfaça as condições do problema
p = 2
# População   Population  Conjunto de indivíduos, ou seja, uma coleção de rotas possíveis
# Seleção     Selection    Baseado no algoritmo do torneio
# Pais        Parents     Combinação de duas rotas para criar uma nova
# Aptidão     Fitness     Função que avalia a melhor rota, no caso, a rota que tem menor distância
# Elitismo    Elitism     Fator que permite passar para próximas gerações indivíduos com melhor desempenho
# Mutação     Mutation    Forma de variar a população trocando de forma aleatória dois pontos em uma rota

def get_data():
    '''Função que carrega os dados salvos no arquivo passado como entrada do problema'''
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("CaixeiroSimples.csv", names=["X", "Y", "Z"])

def createNavigationMap(data: pd.DataFrame, dominio: list[tuple]) -> None:
    '''Função que cria o mapa de navegação no Matplotlib
    Parâmetros
    ----------
    data : pd.DataFrame
        Dados recebidos como entrada do programa
    dominio : list[tuple]
        Limites de domínio
    '''
    x = np.linspace(start=[dominio[0][0], dominio[1][0]], stop=[dominio[0][1], dominio[1][1]], num=1000, axis=1)
    X1, X2 = np.meshgrid(x[0], x[1])
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(data.tail(1).index.item() +1):
        if i == 0:
            ax.scatter(data.at[i, 'X'], data.at[i, 'Y'], data.at[i, 'Z'], marker='*', s=100, linewidth=1, color='green', edgecolors='black', alpha=0.6)
        ax.scatter(data.at[i, 'X'], data.at[i, 'Y'], data.at[i, 'Z'], marker=('*' if i == data.tail(1).index.item() else 'o'), s=(100 if i == data.tail(1).index.item() else 10), linewidth=1, color=('red' if i == data.tail(1).index.item() else 'blue'), edgecolors='black', alpha=0.6)
    ax.set_title('Caixeiro Viajante Tridimensional Simples')
    ax.set_xlabel('Eixo X')
    ax.set_ylabel('Eixo Y')
    ax.set_zlabel('Eixo Z')
    plt.tight_layout()
    ax.view_init(elev=30, azim=-65, roll=0.)
    plt.show()

def tournament():
    return

def selection(population):
    '''Função que seleciona os individuos a serem testados
    '''
    n = rd.randint(0, N)
    best = []
    for i in range(n, N, 2):
        best.append((fitness(np.array(population[i]), np.array(population[i+1])), (population[i], population[i+1])))
        # S = np.concatenate((S, tournament()))
    best.sort()
    return best[:p]

def fitness(p1: np.array, p2: np.array) -> float:
    '''Função para calcular a distância entre os dois pontos no espaço
    Parâmetros
    ----------
    p1 : np.array
        Ponto 1 no espaço
    p2 : np.array
        Ponto 2 no espaço
    Retorno
    -------
    float
        Distância entre os dois pontos no espaço
    '''
    # return np.sqrt(np.sum((p1 - p2) ** 2, axis=0))
    # return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2) + ((p1[2] - p2[2]) ** 2))
    return np.linalg.norm(p1 - p2)

def generatePopulation(data: pd.DataFrame) -> list[tuple]:
    pop = []
    for i in range(data.tail(1).index.item() +1):
        pop.append(
            (data.at[i, 'X'],
             data.at[i, 'Y'],
             data.at[i, 'Z'])
        )
    return pop

def run() -> None:
    '''Execução do programa'''
    data = get_data()
    # limites = [(data.iloc[:, 0].min(), data.iloc[:, 0].max()), (data.iloc[:, 1].min(), data.iloc[:, 1].max())]
    # createNavigationMap(data, limites)
    population = generatePopulation(data)
    # print(population)
    print(selection(population))
    for i in range(10000):
        rankedSolutions = []
        for s in population:
            rankedSolutions.append((fitness(s[i], s[i+1]), s))
        rankedSolutions.sort()
        rankedSolutions.reverse()
        print(f"--- Gen {i} best solution ---")
        print(rankedSolutions[0])
        if rankedSolutions[0][0] > 999:
            break
        bestSolutions = rankedSolutions[:100]
        elements = []
        for s in bestSolutions:
            elements.append(s[1][0])
            elements.append(s[1][1])
            elements.append(s[1][2])
        newGeneration = []
        for _ in range(1000):
            e1 = rd.choice(elements) * rd.uniform(0.99, 1.01)
            e2 = rd.choice(elements) * rd.uniform(0.99, 1.01)
            e3 = rd.choice(elements) * rd.uniform(0.99, 1.01)
            newGeneration.append((e1, e2, e3))
        population = newGeneration

def close() -> None:
    '''Encerramento'''
    sys.exit(0)

try:
    if __name__ == "__main__":
        run()
finally:
    close()
