import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hiperparâmetros:
#
# Pt.inicial  limite inferior do domínio de x.
# Candidato   |x_otimo - x_cand| ≤ ε    # x_cand  ́e um possível candidato da vizinhanca
                                        # ε = 0,1

def perturb(x, e, limites):
    res = np.random.uniform(low=x[0]-e, high=x[1]+e, size=(2, ))
    if res[0] < limites[0][0]:
        res[0] = limites[0][0]
    elif res[0] > limites[0][1]:
        res[0] = limites[0][1]
    if res[1] < limites[1][0]:
        res[1] = limites[1][0]
    elif res[1] > limites[1][1]:
        res[1] = limites[1][1]
    return res

# Domínio da função
# Problema 1
dominio = [(-100, 100), (-100, 100)]
# Problema 2
# dominio = [(-2, 4), (-2, 5)]
# Problema 3
# dominio = [(-8, 8), (-8, 8)]
# Problema 4
# dominio = [(-5.12, 5.12), (-5.12, 5.12)]
# Problema 5
# dominio = [(-10, 10), (-10, 10)]
# Problema 6
# dominio = [(-1, 3), (-1, 3)]
# Problema 7
# dominio = [(0, np.pi), (0, np.pi)]
# Problema 8
# dominio = [(-200, 20), (-200, 20)]

def f(x1, x2):
    # Problema 1 (Minimizar)
    return (x1 ** 2 + x2 ** 2)
    # Problema 2 (Maximizar)
    # return (np.exp(-(x1 ** 2 + x2 ** 2)) + 2 * np.exp(-((x1 - 1.7) ** 2 + (x2 - 1.7) ** 2)))
    # Problema 3 (Minimizar)
    # return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)
    # Problema 4 (Minimizar)
    # return ((x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2 ** 2 - 10 * np.cos(2 * np.pi * x2) + 10))
    # Problema 5 (Maximizar)
    # return (((x1 * np.cos(x1)) / 20) + 2 * np.exp(-(x1 ** 2) - ((x2 - 1) ** 2)) + 0.01 * x1 * x2)
    # Problema 6 (Maximizar)
    # return ((x1 * np.sin(4 * np.pi * x1)) - (x2 * np.sin((4 * np.pi * x2) + np.pi)) + 1)
    # Problema 7 (Minimizar)
    # return ((-np.sin(x1) * np.sin((x1 ** 2)/np.pi) ** (2 * 10)) - (np.sin(x2) * (np.sin((2 * x2 ** 2)/np.pi) ** (2 * 10))))
    # Problema 8 (Minimizar)
    return ((-(x2 + 47)) * np.sin(np.sqrt(np.abs((x1 / 2) + (x2 + 47))))) - (x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47)))))

# Geração do grid e gráfico da função
x = np.linspace(start=[dominio[0][0], dominio[1][0]], stop=[dominio[0][1], dominio[1][1]], num=1000, axis=1)
X1, X2 = np.meshgrid(x[0], x[1])
Y = f(X1, X2)

# Geração do ponto inicial no gráfico
x_otimo = (dominio[0][0], dominio[1][0])
f_otimo = f(x_otimo[0], x_otimo[1])

# Plotagem do desenho do gráfico e do ponto inicial
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
ax.scatter(x_otimo[0], x_otimo[1], f_otimo, marker='x', s=90, linewidth=2, color='red')
# plt.colorbar(surface)

# Etiquetas dos eixos
ax.set_title('f(x1, x2)')
ax.set_xlabel('valores x')
ax.set_ylabel('valores y')
ax.set_zlabel('valores z')
plt.tight_layout()  # Melhor ajuste para a imagem plotada

# Posicionamento da camera de visualização 3d
# Problema 1
ax.view_init(elev=10., azim=-65., roll=0.)
# Problema 2
# ax.view_init(elev=24., azim=-66., roll=0.)
# Problema 3
# ax.view_init(elev=30., azim=-65., roll=0.)
# Problema 4
# ax.view_init(elev=25., azim=-61., roll=0.)
# Problema 5
# ax.view_init(elev=15., azim=-140., roll=0.)
# Problema 6
# ax.view_init(elev=30., azim=-60., roll=0.)
# Problema 7
# ax.view_init(elev=26., azim=-65., roll=0.)
# Problema 8
# ax.view_init(elev=30., azim=160., roll=0.)

# Criação do DataFrame vazio que armazenará as soluções de cada rodada
solucoes = pd.DataFrame(columns = ['x_otimo[0]', 'x_otimo[1]', 'f_otimo'])
for z in range(100):
    # Algoritmo da busca por Hill Climbing
    e = 0.1
    max_iter = 10000
    max_vizinhos = 30
    i = 0
    melhoria = True

    while i < max_iter and melhoria:
        melhoria = False
        for j in range(max_vizinhos):
            x_cand = perturb(x_otimo, e, dominio)
            f_cand = f(x_cand[0], x_cand[1])
            # Minimizar
            if (f_cand < f_otimo):
            # Maximizar
            # if (f_cand > f_otimo):
                melhoria = True
                x_otimo = x_cand
                f_otimo = f_cand
                break
        i+=1
    # Adição da solução encontrada nesta rodada no DataFrame de soluções
    solucoes.loc[z+1] = [x_otimo[0], x_otimo[1], f_otimo]

# Plotagem do gráfico no melhor ponto ótimo
ax.scatter(x_otimo[0], x_otimo[1], f_otimo, marker='o', s=50, linewidth=1, color='green', edgecolors='black')

# Gerando os arquivos com as soluções encontrados (Imagem (PNG) do gráfico com apresentação do ponto inicial e o ponto ótimo e do DataFrame (CSV) de soluções da sequência de 100 rodadas)
# Problema 1
nome_arquivo = 'av1-hc_problema1-funcao1'
# Problema 2
# nome_arquivo = 'av1-hc_problema1-funcao2'
# Problema 3
# nome_arquivo = 'av1-hc_problema1-funcao3'
# Problema 4
# nome_arquivo = 'av1-hc_problema1-funcao4'
# Problema 5
# nome_arquivo = 'av1-hc_problema1-funcao5'
# Problema 6
# nome_arquivo = 'av1-hc_problema1-funcao6'
# Problema 7
# nome_arquivo = 'av1-hc_problema1-funcao7'
# Problema 8
# nome_arquivo = 'av1-hc_problema1-funcao8'
solucoes.to_csv(nome_arquivo+'.csv', index=False)
plt.savefig(nome_arquivo+'.png')

plt.show()