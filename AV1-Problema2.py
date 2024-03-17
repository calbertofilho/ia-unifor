#
# entrada:                                                                              +---+---+---+---+---+---+---+---+
#   - [5, 1, 4, 2, 6, 1, 4, 7]                                                        8 |   |   |   |   |   |   |   |   |
#                                                                                       +---+---+---+---+---+---+---+---+
# identificar onde estão os conflitos no array:                                       7 |   |   |   |   |   |   |   | O |
#   - em toda a coluna do array que se esta navegando                                   +---+---+---+---+---+---+---+---+
#   - em toda a linha que é indicada pelo número salvo no array                       6 |   |   |   |   | O |   |   |   |
#   - e nas duas diagonais (ascendente e descendente) que passam por este ponto         +---+---+---+---+---+---+---+---+
#                                                                                     5 | O |   |   |   |   |   |   |   |
# como representar estes conflitos:                                                     +---+---+---+---+---+---+---+---+
#   - num.array([5, 0, 0, 0, 0, 0, 0, 0])                                             4 |   |   | O |   |   |   | O |   |
#   - num.array([5, 5, 5, 5, 5, 5, 5, 5])                                               +---+---+---+---+---+---+---+---+
#   - num.array([5, 6, 7, 8, 0, 0, 0, 0])                                             3 |   |   |   |   |   |   |   |   |
#   - num.array([5, 4, 3, 2, 1, 0, 0, 0])                                               +---+---+---+---+---+---+---+---+
#                                                                                     2 |   |   |   | O |   |   |   |   |
#   - num.array([0, 0, 4, 0, 0, 0, 0, 0])                                               +---+---+---+---+---+---+---+---+
#   - num.array([4, 4, 4, 4, 4, 4, 4, 4])                                             1 |   | O |   |   |   | O |   |   |
#   - num.array([2, 3, 4, 5, 6, 7, 8, 0])                                               +---+---+---+---+---+---+---+---+
#   - num.array([6, 5, 4, 3, 2, 1, 0, 0])                                                 1   2   3   4   5   6   7   8  
#

import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot

def h(entrada):
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
        # print('Posição da Rainha '+str(pos+1)+' = ('+str(entrada[pos])+', '+str(pos+1)+')')
        # print(rainha)

        movimento_rainha = num.zeros((8, 8), dtype=int)
        diag_ascendente = [-1] * 8
        diag_descendente = [-1] * 8
        for iter in range(len(entrada)):
            movimento_rainha[lin][iter] = 1                          # possibilidade de movimentos para a linha
            movimento_rainha[iter][col] = 1                          # possibilidade de movimentos para a coluna
            if 0 < (entrada[pos] + iter - pos) <= len(entrada): # vetor com a diagonal ascendente
                diag_ascendente[iter] = entrada[pos] + iter - pos
            if 0 < (entrada[pos] - iter + pos) <= len(entrada): # vetor com a diagonal descendente
                diag_descendente[iter] = entrada[pos] - iter + pos
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

    # print('Movimentos da Rainha 5')
    # print(movimentos[4])
        
    # somar o primeiro com:
        # o ultimo
        # o penultimo
        # o antepenultimo
        # até chegar no primeiro
    # verificar as coincidências (valor = 2) e retornar quantas vezes se repetem
    res = 0
    for i in range(len(movimentos)):
        for j in reversed(range(len(movimentos))):
            analise = num.zeros((8, 8), dtype=int)
            if (i < j):
                analise = num.add(rainhas[i], movimentos[j])
                res += num.any(analise[:] == 2)                
    return res

if __name__ == '__main__':
    # p.161-165
    exemplo_livro1 = num.array([2, 4, 7, 4, 8, 5, 5, 2], dtype=int)
    exemplo_livro2 = num.array([3, 2, 7, 5, 2, 4, 1, 1], dtype=int)
    exemplo_livro3 = num.array([2, 4, 4, 1, 5, 1, 2, 4], dtype=int)
    exemplo_livro4 = num.array([3, 2, 5, 4, 3, 2, 1, 3], dtype=int)

    teste =  num.array([5, 1, 4, 2, 6, 1, 4, 7], dtype=int)
    print(h(entrada=teste))
