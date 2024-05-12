import sys
import numpy as np
import matplotlib.pyplot as plt

# conversão: 10 base₁₀  ↔  1010 base₂
A = 10                  #constante
populationSize = 1000   #N  → Tamanho da população
bitsSequence = 4        #nd → Quantidade de bits presentes para cada parâmetro p
variablesCount = 20     #p  → Quantidade de parâmetros do problema
limitsMin = -10         #l  → Limite inferior do domínio
limitsMax = 10          #u  → Limite superior do domínio
numberOfRandomFlips = 5 # Número máximo de posiçòes que podem ser trocadas

def plot() -> None:
    limites = [(-10, 10)] * 2
    def funcao(x1, x2):
        return ((np.power(x1, 2) - 10 * np.cos(2 * np.pi * x1) + 10) + (np.power(x2, 2) - 10 * np.cos(2 * np.pi * x2) + 10))
    # Geração do grid e gráfico da função
    x = np.linspace(start=[limites[0][0], limites[1][0]], stop=[limites[0][1], limites[1][1]], num=1000, axis=1)
    X1, X2 = np.meshgrid(x[0], x[1])
    Y = funcao(X1, X2)
    # Plotagem do desenho do gráfico e do ponto inicial
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
    plt.colorbar(surface)
    # Etiquetas dos eixos
    ax.set_title("rastrigin")
    ax.set_xlabel('valores x')
    ax.set_ylabel('valores y')
    ax.set_zlabel('valores z')
    plt.tight_layout()  # Melhora o ajuste para a imagem a ser plotada
    ax.view_init(elev=10, azim=-65, roll=0.)
    plt.show()

def phi(number):
    decode = 0
    for i in range(len(number)):
        decode += np.power(number[(len(number) - 1 - i)] * 2, i)
    return (limitsMin + (((limitsMax - limitsMin) / (np.power(2, bitsSequence) - 1)) * decode))

def getNumbers(individual):
    numbers = []
    for i in range(0, len(individual), bitsSequence):
        numbers.append(phi(individual[i:(i + bitsSequence)]))
    return numbers

def f(x):
    ################################################
    # Rastrigin
    # 
    # f(ꭓ) = A · p + Ʃ֡ᵢ₌₁ (ꭓᵢ² − A · cos(2 · π · ꭓᵢ))
    ################################################
    f = A * len(x)
    for i in range(len(x)):
        f += np.power(x[i], 2) - A * np.cos(2 * np.pi * x[i])
    return f

def fitness(individual):
    return f(getNumbers(individual)) + 1

def generatePopulation():
    return np.random.randint(low=0, high=2, size=(populationSize, (variablesCount * bitsSequence)))

def rouletteWheel(population):
    roulette = []
    totalFitness = 0
    maxFitness = fitness(population[0])
    for i in range(len(population)):
        if fitness(population[i]) > maxFitness:
            maxFitness = fitness(population[i])
    maxFitness = maxFitness + 1
    for i in range(len(population)):
        totalFitness += (maxFitness - fitness(population[i]))
    point = 0
    for i in range(len(population)):
        roulette.append(((maxFitness - fitness(population[i])) / totalFitness) + point)
        point = roulette[i]
    return roulette

def selection(population, roulette):
    randRoulette = np.random.uniform()
    for i in range(len(population)):
        if roulette[i] > randRoulette:
            return np.copy(population[i])

def crossOver(parent1, parent2, numberOfCuts):
    cuts = []
    for i in range(numberOfCuts):
        cuts.append(np.random.randint(len(parent1)))
    cuts.sort()
    child1 = parent1[0:cuts[0]]
    child2 = parent2[0:cuts[0]]
    for i in range(1, numberOfCuts):
        if i % 2 == 0:
            child1 = child1 + parent1[cuts[i-1]:cuts[i]]
            child2 = child2 + parent2[cuts[i-1]:cuts[i]]
        else:
            child1 = child1 + parent2[cuts[i-1]:cuts[i]]
            child2 = child2 + parent1[cuts[i-1]:cuts[i]]
    if numberOfCuts % 2 == 0:
        child1 = child1 + parent1[cuts[numberOfCuts-1]:]
        child2 = child2 + parent2[cuts[numberOfCuts-1]:]
    else:
        child1 = child1 + parent2[cuts[numberOfCuts-1]:]
        child2 = child2 + parent1[cuts[numberOfCuts-1]:]
    return child1, child2

def sortPopulation(population):
    return sorted(population, key=lambda x: fitness(x), reverse=False)

def mutateFlipBit(individual):
    individualList = list(individual)
    for flips in range(np.random.randint(numberOfRandomFlips)):
        randFlip = np.random.randint(len(individual))
        individualList[randFlip] = 1 if (individualList[randFlip] == 0) else 0
    return np.copy(individualList)

def run() -> None:
#   1. Criação da população inicial
    population = generatePopulation()
#   2. Seleção para procriaçáo
    roulette = rouletteWheel(population)
    selectedIndividual = selection(population, roulette)
    print(selectedIndividual)
#   3. Procriação
    sortedPopulation = sortPopulation(population)
#   4. Mutação
    mutatetedIndividual = mutateFlipBit(selectedIndividual)
    print(mutatetedIndividual)










#  https://github.com/CelsoMeireles/Rastrigin-Function-Genetic-Algorithm/blob/main/Rastrigin%20Function%20Genetic%20Algorithm.ipynb


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

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        # plot()
        run()
finally:
    close()
