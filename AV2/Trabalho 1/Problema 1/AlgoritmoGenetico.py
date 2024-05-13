import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# conversão: 10 base₁₀  ↔  1010 base₂
A = 10                   # constante
maxGenerations = 100     # 
populationSize = 30      #N  → Tamanho da população
bitsSequence = 4         #nd → Quantidade de bits presentes para cada parâmetro p
variablesCount = 20      #p  → Quantidade de parâmetros do problema
limitsMin = -10          #l  → Limite inferior do domínio
limitsMax = 10           #u  → Limite superior do domínio
chanceToRecombine = 0.85 #pr → Probabilidade de recombinação
chanceToMutate = 0.01    #pm → Probabilidade de mutar
cutMask = 1         #mascara → Número de cortes na sequencia dos bits
numberOfRandomFlips = 5  # Número máximo de posiçòes que podem ser trocadas

def load_full_data() -> pd.DataFrame:
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    return pd.read_csv("fitnessMatrix.csv", names=['min', 'max', 'sum', 'mean', 'std'], skiprows=[0]) # no header

def check_dirs() -> None:
    # verifica e cria os diretorios necessarios para salvar os arquivos
    if not os.path.exists('AV2/Trabalho 1/Problema 1'):
        os.makedirs('AV2/Trabalho 1/Problema 1')
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

def plot(savefig: bool) -> None:
    def funcao(x1, x2):
        return ((np.power(x1, 2) - 10 * np.cos(2 * np.pi * x1) + 10) + (np.power(x2, 2) - 10 * np.cos(2 * np.pi * x2) + 10))
    # Geração do grid e gráfico da função
    limites = [(-10, 10)] * 2
    x = np.linspace(start=[limites[0][0], limites[1][0]], stop=[limites[0][1], limites[1][1]], num=1000, axis=1)
    X1, X2 = np.meshgrid(x[0], x[1])
    Y = funcao(X1, X2)
    # Plotagem do desenho do gráfico e do ponto inicial
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection='3d')
    surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
    plt.colorbar(surface)
    # Etiquetas dos eixos
    ax.set_title("Rastrigin")
    ax.set_xlabel('valores x')
    ax.set_ylabel('valores y')
    ax.set_zlabel('valores z')
    plt.tight_layout()  # Melhora o ajuste para a imagem a ser plotada
    ax.view_init(elev=10, azim=-65, roll=0.)
    if savefig:
        plt.savefig('funcao.png')
    plt.show()
    return ax

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

def calculateFitness(population) -> None:
    fit = []
    for individual in population:
        fit.append(fitness(individual))
    minimum = min(fit)
    maximum = max(fit)
    amount = np.sum(fit)
    average = np.mean(fit)
    stdDeviation = np.std(fit)
    return minimum, maximum, amount, average, stdDeviation

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
    child1 = np.array(parent1[0:cuts[0]])
    child2 = np.array(parent2[0:cuts[0]])
    for i in range(1, numberOfCuts):
        if i % 2 == 0:
            child1 = np.append(child1, np.array(parent1[cuts[i-1]:cuts[i]]))
            child2 = np.append(child2, np.array(parent2[cuts[i-1]:cuts[i]]))
        else:
            child1 = np.append(child1, np.array(parent2[cuts[i-1]:cuts[i]]))
            child2 = np.append(child2, np.array(parent1[cuts[i-1]:cuts[i]]))
    if numberOfCuts % 2 == 0:
        child1 = np.append(child1, np.array(parent1[cuts[numberOfCuts-1]:]))
        child2 = np.append(child2, np.array(parent2[cuts[numberOfCuts-1]:]))
    else:
        child1 = np.append(child1, np.array(parent2[cuts[numberOfCuts-1]:]))
        child2 = np.append(child2, np.array(parent1[cuts[numberOfCuts-1]:]))
    return child1, child2

def sortPopulation(population):
    return sorted(population, key=lambda x: fitness(x), reverse=False)

def mutateFlipBit(individual):
    individualList = list(individual)
    for flips in range(np.random.randint(numberOfRandomFlips)):
        randFlip = np.random.randint(len(individual))
        individualList[randFlip] = 1 if (individualList[randFlip] == 0) else 0
    return np.copy(individualList)

def printProgressBar(value: float, label: str) -> None:
    animation1 = '|/-\\'
    animation2 = [
        "[-     ]",
        "[ -    ]",
        "[  -   ]",
        "[   -  ]",
        "[    - ]",
        "[     -]",
        "[    - ]",
        "[   -  ]",
        "[  -   ]",
        "[ -    ]"
    ]
    animation3 = [
        "[        ]",
        "[-       ]",
        "[--      ]",
        "[---     ]",
        "[----    ]",
        "[-----   ]",
        "[------  ]",
        "[------- ]",
        "[--------]",
        "[ -------]",
        "[  ------]",
        "[   -----]",
        "[    ----]",
        "[     ---]",
        "[      --]",
        "[       -]"
    ]
    animation4 = [
        "    ",
        "░   ",
        "▒   ",
        "▓   ",
        "█   ",
        "█░  ",
        "█▒  ",
        "█▓  ",
        "██  ",
        "██░ ",
        "██▒ ",
        "██▓ ",
        "███ ",
        "███░",
        "███▒",
        "███▓",
        "████",
        "▓███",
        "▒███",
        "░███",
        " ███",
        " ▓██",
        " ▒██",
        " ░██",
        "  ██",
        "  ▓█",
        "  ▒█",
        "  ░█",
        "   █",
        "   ▓",
        "   ▒",
        "   ░"
    ]
    animation = animation3
    max = 100
    j = value / max
    sys.stdout.write('\r')
    sys.stdout.write(f"{label.ljust(10)} {animation[int(100 * j) % len(animation) if int(100 * j) != 100 else 8]} {int(100 * j)}% ")
    sys.stdout.flush()

def run(save: bool) -> None:
    fitnessMatrix = np.empty((maxGenerations, 5))
#   1. Geração da população inicial aleatória
    population = generatePopulation()
    for generation in range(maxGenerations):
        printProgressBar((generation / maxGenerations) * 100, 'Calculando...')
#       2. Ordenação da população, dessa geração, segundo as aptidões dos indivíduos
        population = sortPopulation(population)
#       3. Cálculo das aptidões e criação da tabela com as informações solicitadas
        fitnessMatrix[generation] = np.array(calculateFitness(population))
        newPopulation = []
        for _ in range(0, populationSize, 2):
#           4. Seleção para procriaçáo
            roulette = rouletteWheel(population)
            parent1 = selection(population, roulette)
            newIndividual1 = np.copy(parent1)
            parent2 = selection(population, roulette)
            newIndividual2 = np.copy(parent2)
#           5. Procriação
            if np.random.uniform() <= chanceToRecombine:
                newIndividual1, newIndividual2 = crossOver(parent1=parent1, parent2=parent2, numberOfCuts=cutMask)
            newPopulation.append(newIndividual1)
            newPopulation.append(newIndividual2)
#       6. Mutação
        for individual in newPopulation:
            if np.random.uniform() <= chanceToMutate:
                individual = mutateFlipBit(individual)
        population = newPopulation
    printProgressBar(100, 'Concluído !!!')
    df = pd.DataFrame(data=fitnessMatrix, columns=['min', 'max', 'sum', 'mean', 'std'])
    posMin = gen = np.argmin(fitnessMatrix[:, 0])
    posMax = np.argmax(fitnessMatrix[:, 1])
    msg = "\n"
    msg += (f"Solução da função de Rastrigin encontrada com aptidão de {fitness(population[0]):.4f}, a partir da {gen+1}ª geração\n\n")
    msg += (f"Menor aptidão: {fitnessMatrix[posMin, 0]:.4f}\n")
    msg += (f"Maior aptidão: {fitnessMatrix[posMax, 1]:.4f}\n")
    msg += (f"Média de todas aptidões: {fitnessMatrix[:, 3].mean():.4f}\n")
    msg += (f"Desvio padrão das aptidões em todas as rodadas: {fitnessMatrix[:, 4].std():.4f}\n")
    if save:
        # cria arquivo de texto com os resultados dos cálculos
        with open('resultado.txt', 'w') as arquivo:
            arquivo.write(msg)
        df.to_csv('fitnessMatrix.csv', index=False)
    print(msg)

def evolution(data: pd.DataFrame, savefig: bool) -> None:
    # Plot evolution
    df_reversed = data[::-1]
    plt.plot(df_reversed['min'])
    plt.title('Evolução das gerações')
    plt.ylabel('Aptidão: Rastrigin f(x)')
    plt.xlabel('Gerações')
    if savefig:
        plt.savefig('evolucao.png')
    plt.show()

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        check_dirs()
        plot(True)
        run(True)
        evolution(load_full_data(), True)
finally:
    close()
