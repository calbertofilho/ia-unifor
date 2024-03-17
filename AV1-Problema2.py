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

entrada = num.array([5, 1, 4, 2, 6, 1, 4, 7], dtype=int)
matriz = num.zeros((8, 8), dtype=int)
for posicao in range(len(entrada)):
    coluna = posicao
    linha = len(entrada) - entrada[posicao]
    matriz[linha][coluna] = 1
print('Entrada')
print(matriz)
print('')

for pos in range(len(entrada)):
    col = pos
    lin = len(entrada) - entrada[pos]

    rainha = num.zeros((8, 8), dtype=int)
    rainha[lin][col] = 1

    movimentos = num.zeros((8, 8), dtype=int)
    for iter in range(len(entrada)):
        movimentos[lin][iter] = 1    # linha
        movimentos[iter][col] = 1    # coluna
        print(str(lin)+' '+str(col))
        # num.fill_diagonal(movimentos[-lin:, -col:], 9) # diag ascendente
        num.fill_diagonal(movimentos[lin:, col:], 5) # diag descendente

    print('Posição da Rainha '+str(pos+1)+' = ('+str(entrada[pos])+', '+str(pos+1)+')')
    print(rainha)
    print('Movimentos da Rainha '+str(pos+1))
    print(movimentos)
    # print('')
    # print(num.identity(8, dtype=int))
    # print(num.fliplr(num.identity(8, dtype=int)))
