#
# entrada:                                                                           +---+---+---+---+---+---+---+---+
#   - [5, 1, 4, 2, 6, 1, 4, 7]                                                     8 |   |   |   |   |   |   |   |   |
#                                                                                    +---+---+---+---+---+---+---+---+
# identificar onde estão os conflitos no array:                                    7 |   |   |   |   |   |   |   | O |
#   - em toda a coluna do array que se esta navegando                                +---+---+---+---+---+---+---+---+
#   - em toda a linha que é indicada pelo número salvo no array                    6 |   |   |   |   | O |   |   |   |     [ 5   1   4   2   6   1   4   7 ]
#   - e nas duas diagonais (ascendente e descendente) que passam por este ponto      +---+---+---+---+---+---+---+---+
#                                                                                  5 | O |   |   |   |   |   |   |   |    h(x) = qntde de rainhas q se atacam 
# como representar estes conflitos:                                                  +---+---+---+---+---+---+---+---+    h(x) = 7 (implementar essa função)
#   - num.array([5, 0, 0, 0, 0, 0, 0, 0])  Rainha 1                                4 |   |   | O |   |   |   | O |   |
#   - num.array([5, 5, 5, 5, 5, 5, 5, 5])                                            +---+---+---+---+---+---+---+---+    f(x) = 28 - h(x)
#   - num.array([5, 6, 7, 8, 0, 0, 0, 0])                                          3 |   |   |   |   |   |   |   |   |    f(x) = 28 - 7
#   - num.array([5, 4, 3, 2, 1, 0, 0, 0])                                            +---+---+---+---+---+---+---+---+    f(x) = 21 (solução deste exemplo)
#                                                                                  2 |   |   |   | O |   |   |   |   |
#   - num.array([0, 0, 4, 0, 0, 0, 0, 0])  Rainha 3                                  +---+---+---+---+---+---+---+---+
#   - num.array([4, 4, 4, 4, 4, 4, 4, 4])                                          1 |   | O |   |   |   | O |   |   |
#   - num.array([2, 3, 4, 5, 6, 7, 8, 0])                                            +---+---+---+---+---+---+---+---+
#   - num.array([6, 5, 4, 3, 2, 1, 0, 0])                                              1   2   3   4   5   6   7   8  
#

import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot
import timeit
import warnings
import sys

# não exibir os avisos
warnings.filterwarnings('ignore')

def f(entrada):
    # # Matriz recebida como entrada
    # matriz = num.zeros((8, 8), dtype=int)
    # for posicao in range(len(entrada)):
    #     coluna = posicao
    #     linha = len(entrada) - entrada[posicao]
    #     matriz[linha][coluna] = 1
    # print('Entrada')
    # print(matriz)
    # print('')

    rainhas = [0] * len(entrada)
    movimentos = [0] * len(entrada)

    for pos in range(len(entrada)):
        col = pos
        lin = len(entrada) - entrada[pos]

        # Posicao da rainha que esta sendo analisada
        rainha = num.zeros((8, 8), dtype=int)
        rainha[lin][col] = 1
        rainhas[pos] = rainha
        # print('Posição da Rainha %i = (%i, %i)' % pos+1, pos, pos+1)
        # print(rainha)

        movimento_rainha = num.zeros((8, 8), dtype=int)
        diag_ascendente = [-1] * 8
        diag_descendente = [-1] * 8
        for iter in range(len(entrada)):
            movimento_rainha[lin][iter] = 1                          # possibilidade de movimentos para a linha
            movimento_rainha[iter][col] = 1                          # possibilidade de movimentos para a coluna
            if 0 < (entrada[pos] + iter - pos) <= len(entrada):
                diag_ascendente[iter] = entrada[pos] + iter - pos  # vetor com a diagonal ascendente
            if 0 < (entrada[pos] - iter + pos) <= len(entrada):
                diag_descendente[iter] = entrada[pos] - iter + pos # vetor com a diagonal descendente
        for i in range(len(diag_ascendente)):                        # possibilidade de movimentos para a diagonal ascendente
            column = i
            if not(diag_ascendente[i] == -1):
                row = len(diag_ascendente) - diag_ascendente[i]
                movimento_rainha[row][column] = 1
        for j in range(len(diag_descendente)):                       # possibilidade de movimentos para a diagonal descendente
            c = j
            if not(diag_descendente[j] == -1):
                r = len(diag_descendente) - diag_descendente[j]
                movimento_rainha[r][c] = 1

        movimentos[pos] = movimento_rainha

    # somar a matriz de posição da primeira rainha com a matriz de possibilidades da:
        # última rainha
        # penúltima rainha
        # antepenúltima rainha
        # até chegar na segunda rainha
    # verificar as coincidências (valor = 2) e retornar quantas vezes se repetem
    res = 0
    for i in range(len(movimentos)):
        for j in reversed(range(len(movimentos))):
            analise = num.zeros((8, 8), dtype=int)
            if (i < j):
                analise = num.add(rainhas[i], movimentos[j])
                res += num.any(analise[:] == 2)                

    return (28 - res)

def perturb(x):
    res = []
    # procurar e eliminar os repetidos trocando o valor dos elementos
    if len(set(x)) != len(x):
        for elem in x:
            if (elem in res):
                novo_elem = rd.randint(1, 8)
                while ((novo_elem in res) or (novo_elem in x)):
                    novo_elem = rd.randint(1, 8)
                res.append(novo_elem)
            else:
                res.append(elem)
                elem = 0
        print('perturb = %s' % str(num.array(res, dtype=int)))
        sys.exit()
        return num.array(res, dtype=int)
    else:
        res = x
    # fazer a mudança de local dos elementos executando uma lista circular de três posições, a partir do index sorteado num rd.randint(0, 7)
    pos = rd.randint(0, 7)
    if pos == 7:
        pos1 = pos
        pos2 = 0
        pos3 = 1
    elif pos == 6:
        pos1 = pos
        pos2 = pos + 1
        pos3 = 0
    else:
        pos1 = pos
        pos2 = pos + 1
        pos3 = pos + 2
    res[pos1], res[pos2], res[pos3] = res[pos3], res[pos1], res[pos2]
    return res

def otimizar(posicoes):
    entrada = posicoes
 
    x_otimo = entrada
    print('entrada = %s' % str(x_otimo))
    f_otimo = f(x_otimo)
    temp = 100
    decaimento = 0.97

    for _ in range(1000):
        x_cand = perturb(x_otimo)
        f_cand = f(x_cand)
        P_ij = num.exp(-(f_cand - f_otimo) / temp)
        if ((f_cand > f_otimo) or (P_ij >= num.random.uniform(0, 1))):
            x_otimo = x_cand
            f_otimo = f_cand
        temp *= decaimento

    return x_otimo

if __name__ == '__main__':
    # entradas retiradas do livro-texto (p.161-165)
    exemplo_livro1 = num.array([2, 4, 7, 4, 8, 5, 5, 2], dtype=int)
    exemplo_livro2 = num.array([3, 2, 7, 5, 2, 4, 1, 1], dtype=int)
    exemplo_livro3 = num.array([2, 4, 4, 1, 5, 1, 2, 4], dtype=int)
    exemplo_livro4 = num.array([3, 2, 5, 4, 3, 2, 1, 3], dtype=int)

    # soluções possíveis
    solucao1 = num.array([2, 4, 6, 8, 3, 1, 7, 5], dtype=int)
    solucao2 = num.array([8, 2, 4, 1, 7, 5, 3, 6], dtype=int)

    # entrada passada no trabalho
    teste =  num.array([5, 1, 4, 2, 6, 1, 4, 7], dtype=int)

    inicio = timeit.default_timer()
    sol = otimizar(posicoes=teste)
    print('solucao = %s' % str(sol))  # A soluçáo retornada ainda não é válida !!!
    fim = timeit.default_timer()

    print(f"Tempo de execução: {fim - inicio:.4f} segundos")
