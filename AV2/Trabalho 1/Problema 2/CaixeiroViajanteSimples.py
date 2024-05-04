# DEFINIÇÕES
# Gene       Gene        Ponto no espaço representado por coordenadas (x, y, z)
# Indivíduo  Individual  Rota única que satisfaça as condições do problema (distância euclidiana entre dois pontos no espaço)
# População  Population  Conjunto de indivíduos, ou seja, uma coleção de rotas possíveis
# Gerações   Generation  Quantidade máxima de gerações
# Seleção    Selection   Baseado no algoritmo do torneio
# Pais       Parents     Combinação de duas rotas para criar uma nova
# Aptidão    Fitness     Função que avalia a melhor rota, no caso, a rota que tem menor distância
# Elitismo   Elitism     Fator que permite passar para próximas gerações indivíduos com melhor desempenho
# Mutação    Mutation    Forma de variar a população trocando, aleatoriamente, dois pontos em uma rota
# 
# 
# PASSOS
# 1. Leitura dos dados
# 2. Criação da população inicial
# 3. Seleção dos pais para procriaçáo
# 4. Procriação
# 5. Mutação
# 6. Elitismo
# 7. Repetição do ciclo por n gerações
import os
import sys
import math
import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

N = 100         # Quantidade de coordenadas no espaço
p = 2           # Quantidade de parâmetros do problema, neste caso, dois pontos do espaço para definirmos a distância entre eles
maxGen = 1000   # Quantidade máxima de gerações

def get_data() -> pd.DataFrame:
    ''' Função que carrega os dados salvos no arquivo passado como entrada do problema
    Parâmetros
    ----------
    None
    Retorno
    -------
    pd.DataFrame
        Dados coletados '''
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("CaixeiroSimples.csv", names=["X", "Y", "Z"])

def createNavigationMap(data: pd.DataFrame) -> None:
    ''' Função que cria o mapa de navegação no Matplotlib
    Parâmetros
    ----------
    data : pd.DataFrame
        Dados recebidos como entrada do programa '''
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

def plot(data) -> None:
    ''' Função que plota o gráfico no Matplotlib '''
    createNavigationMap(data)
    X = (data.iloc[:, 0].min(), data.iloc[:, 0].max())
    Y = (data.iloc[:, 1].min(), data.iloc[:, 1].max())
    Z = (data.iloc[:, 2].min(), data.iloc[:, 2].max())
    realm = [X, Y, Z]

def generatePopulation(data: pd.DataFrame) -> np.ndarray:
    ''' Função que define a população inicial
    Parâmetros
    ----------
    data : pd.DataFrame
        Dados catalogados
    Retorno
    -------
    np.ndarray
        População inicial '''
    initialPopulation = np.empty(shape=[0, 3])
    for i in range(data.tail(1).index.item() + 1):
        initialPopulation = np.append(initialPopulation, [(data.at[i, 'X'], data.at[i, 'Y'], data.at[i, 'Z'])], axis=0)
    return initialPopulation

def fitness(p1: np.ndarray, p2: np.ndarray) -> float:
    ''' Função para calcular a distância euclidiana entre dois pontos num espaço n-dimensional:
            dist(A, B) = √((B1 - A1)² + (B2 - A2)² + ... + (Bn - An)²)
    Parâmetros
    ----------
    p1 : np.ndarray
        Ponto 1 no espaço
    p2 : np.ndarray
        Ponto 2 no espaço
    Retorno
    -------
    float
        Valor real com a distância dos dois pontos '''
    # sum = 0
    # if p1.size == p2.size:
    #     for i in range(p1.size):
    #         sum += (p2[i] - p1[i]) ** 2
    # return math.sqrt(sum)
    # return np.sqrt(np.sum((p1 - p2) ** 2, axis=0))
    return np.linalg.norm(p1 - p2)

def tournament(population: np.ndarray) -> list[tuple]:
    ''' Função que seleciona um grupo de n indivíduos aleatórios da população (em que n < N) e o desempenho dos n indivíduos são avaliados
    Parâmetros
    ----------
    population : np.array
        População
    Retorno
    -------
    list[tuple]
        Seleção ranqueada das rotas avaliadas '''
    n = rd.randint(1, N)
    rankedSolutions = []
    for i in range(n, N, p):
        rankedSolutions.append((fitness(population[i], population[i+1]), (population[i], population[i+1])))
    rankedSolutions.sort(key=lambda x: x[0])
    return rankedSolutions

def selection(population: np.ndarray) -> tuple:
    ''' Função que seleciona os individuos a serem combinados
    Parâmetros
    ----------
    population : np.array
        População
    Retorno
    -------
    tuple
        Seleção dos pais para a procriação '''
    best = tournament(population)[:1]
    return best

def recombination(parents):
    ...

def run() -> None:
    ''' Execução do programa '''
#   1. Leitura dos dados
    data = get_data()
#   2. Criação da população inicial
    population = generatePopulation(data)
    for _ in range(maxGen):
#       3. Seleção dos pais para procriaçáo
        parents = selection(population)
        print(parents)
#       4. Procriação
#       5. Mutação
#       6. Elitismo
#   7. Repetição do ciclo por 'maxGen' gerações

def close() -> None:
    ''' Encerramento '''
    sys.exit(0)

try:
    if __name__ == "__main__":
        plot(get_data())
        run()
finally:
    close()
