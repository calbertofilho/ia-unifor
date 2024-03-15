# Hiperparâmetros:
#
# Pt.inicial   limite inferior do domínio de x.
# Candidato    |x_otimo - x_cand| ≤ ε      x_cand ́e um possível candidato da vizinhanca
# ε = 0,1

import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

# Problema 1 (Minimizar)
tipo1 = 'min'
limite1 = [(-100, 100), (-100, 100)]
def funcao1(x1, x2):
    return (x1 ** 2 + x2 ** 2)
# Problema 2 (Maximizar)
tipo2 = 'max'
limite2 = [(-2, 4), (-2, 5)]
def funcao2(x1, x2):
    return (np.exp(-(x1 ** 2 + x2 ** 2)) + 2 * np.exp(-((x1 - 1.7) ** 2 + (x2 - 1.7) ** 2)))
# Problema 3 (Minimizar)
tipo3 = 'min'
limite3 = [(-8, 8), (-8, 8)]
def funcao3(x1, x2):
    return (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)
# Problema 4 (Minimizar)
tipo4 = 'min'
limite4 = [(-5.12, 5.12), (-5.12, 5.12)]
def funcao4(x1, x2):
    return ((x1 ** 2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2 ** 2 - 10 * np.cos(2 * np.pi * x2) + 10))
# Problema 5 (Maximizar)
tipo5 = 'max'
limite5 = [(-10, 10), (-10, 10)]
def funcao5(x1, x2):
    return (((x1 * np.cos(x1)) / 20) + 2 * np.exp(-(x1 ** 2) - ((x2 - 1) ** 2)) + 0.01 * x1 * x2)
# Problema 6 (Maximizar)
tipo6 = 'max'
limite6 = [(-1, 3), (-1, 3)]
def funcao6(x1, x2):
    return ((x1 * np.sin(4 * np.pi * x1)) - (x2 * np.sin((4 * np.pi * x2) + np.pi)) + 1)
# Problema 7 (Minimizar)
tipo7 = 'min'
limite7 = [(0, np.pi), (0, np.pi)]
def funcao7(x1, x2):
    return ((-np.sin(x1) * np.sin((x1 ** 2)/np.pi) ** (2 * 10)) - (np.sin(x2) * (np.sin((2 * x2 ** 2)/np.pi) ** (2 * 10))))
# Problema 8 (Minimizar)
tipo8 = 'min'
limite8 = [(-200, 20), (-200, 20)]
def funcao8(x1, x2):
    return ((-(x2 + 47)) * np.sin(np.sqrt(np.abs((x1 / 2) + (x2 + 47))))) - (x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47)))))

def perturb_hc(x, e, limites):
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

def perturb_grs(x, limites):
    res = np.random.uniform(low=x[0]*rd.random(), high=x[1]*rd.random(), size=(2, ))
    if res[0] < limites[0][0]:
        res[0] = limites[0][0]
    elif res[0] > limites[0][1]:
        res[0] = limites[0][1]
    if res[1] < limites[1][0]:
        res[1] = limites[1][0]
    elif res[1] > limites[1][1]:
        res[1] = limites[1][1]
    return res

def otimizar(tipo, busca, f, dominio, x_otimo, abert, max_it, max_viz, pos, salvar):
    # Criação do DataFrame vazio que armazenará as soluções de cada rodada
    solucoes = pd.DataFrame(columns = ['pt_otimo[x]', 'pt_otimo[y]', 'f(otimo)'])

    # Geração do grid e gráfico da função
    x = np.linspace(start=[dominio[0][0], dominio[1][0]], stop=[dominio[0][1], dominio[1][1]], num=1000, axis=1)
    X1, X2 = np.meshgrid(x[0], x[1])
    Y = f(X1, X2)

    # Geração do ponto inicial no gráfico
    f_otimo = f(x_otimo[0], x_otimo[1])

    # Plotagem do desenho do gráfico e do ponto inicial
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
    ax.scatter(x_otimo[0], x_otimo[1], f_otimo, marker='x', s=90, linewidth=2, color='red')
    # plt.colorbar(surface)

    # Etiquetas dos eixos
    ax.set_title(f.__name__)
    ax.set_xlabel('valores x')
    ax.set_ylabel('valores y')
    ax.set_zlabel('valores z')
    plt.tight_layout()  # Melhora o ajuste para a imagem a ser plotada

    for z in range(100):
        # Algoritmo da busca por Hill Climbing
        if (busca == 'hc'):
            e = abert
            max_iter = max_it
            max_vizinhos = max_viz
            i = 0
            melhoria = True

            while i < max_iter and melhoria:
                melhoria = False
                for _ in range(max_vizinhos):
                    x_cand = perturb_hc(x_otimo, e, dominio)
                    f_cand = f(x_cand[0], x_cand[1])
                    if (tipo == 'min'): # Minimizar
                        teste = (f_cand < f_otimo)
                    elif (tipo == 'max'): # Maximizar
                        teste = (f_cand > f_otimo)
                    if teste:
                        melhoria = True
                        x_otimo = x_cand
                        f_otimo = f_cand
                        break
                i += 1
        # Algoritmo da busca por Local Random Search
        elif (busca == 'lrs'):
            ... #https://freedium.cfd/https://medium.com/analytics-vidhya/how-does-random-search-algorithm-work-python-implementation-b69e779656d6
        # Algoritmo da busca por Global Random Search
        elif (busca == 'grs'):
            max = max_it
            i = 0
            while i < max:
                x_cand = perturb_grs(x_otimo, dominio)
                f_cand = f(x_cand[0], x_cand[1])
                if (tipo == 'min'): # Minimizar
                    teste = (f_cand < f_otimo)
                elif (tipo == 'max'): # Maximizar
                    teste = (f_cand > f_otimo)
                if teste:
                    x_otimo = x_cand
                    f_otimo = f_cand
                    break
                i += 1

        # Adição da solução encontrada nesta rodada no DataFrame de soluções
        solucoes.loc[z+1] = [x_otimo[0], x_otimo[1], f_otimo]

    # Plotagem do gráfico no melhor ponto ótimo
    ax.scatter(x_otimo[0], x_otimo[1], f_otimo, marker='o', s=50, linewidth=1, color='green', edgecolors='black')

    # Posicionamento da camera de visualização 3d
    # ax.view_init(pos)
    if (pos == 'view1'):
        ax.view_init(elev=10., azim=-65., roll=0.)
    elif (pos == 'view2'):
        ax.view_init(elev=24., azim=-66., roll=0.)
    elif (pos == 'view3'):
        ax.view_init(elev=30., azim=-65., roll=0.)
    elif (pos == 'view4'):
        #ax.view_init(elev=25., azim=-61., roll=0.)
        ax.view_init(elev=6., azim=-62., roll=0.)
    elif (pos == 'view5'):
        #ax.view_init(elev=15., azim=-140., roll=0.)
        ax.view_init(elev=35., azim=-91., roll=0.)
    elif (pos == 'view6'):
        ax.view_init(elev=30., azim=-60., roll=0.)
    elif (pos == 'view7'):
        ax.view_init(elev=26., azim=-65., roll=0.)
    elif (pos == 'view8'):
        #ax.view_init(elev=30., azim=160., roll=0.)
        ax.view_init(elev=18., azim=-125., roll=0.)

    # Gerando os arquivos com as soluções encontrados (Imagem (PNG) do gráfico com apresentação do ponto inicial e o ponto ótimo e do DataFrame (CSV) de soluções da sequência de 100 rodadas)
    if salvar:
        nome_arquivo = 'av1-'+busca+'_'+f.__name__
        solucoes.to_csv(nome_arquivo+'.csv', index=False)
        plt.savefig(nome_arquivo+'.png')

    plt.show()

if __name__ == '__main__':
    otimizar(tipo=tipo1, busca='hc', f=funcao1, dominio=limite1, x_otimo=(limite1[0][0], limite1[1][0]), abert=0.10, max_it=1000, max_viz=30, pos='view1', salvar=False)
    otimizar(tipo=tipo2, busca='hc', f=funcao2, dominio=limite2, x_otimo=(limite2[0][0], limite2[1][0]), abert=0.10, max_it=1000, max_viz=30, pos='view2', salvar=False)
    otimizar(tipo=tipo3, busca='hc', f=funcao3, dominio=limite3, x_otimo=(limite3[0][0], limite3[1][0]), abert=0.10, max_it=1000, max_viz=30, pos='view3', salvar=False)
    otimizar(tipo=tipo4, busca='hc', f=funcao4, dominio=limite4, x_otimo=(limite4[0][0], limite4[1][0]), abert=0.10, max_it=1000, max_viz=30, pos='view4', salvar=False)
    otimizar(tipo=tipo5, busca='hc', f=funcao5, dominio=limite5, x_otimo=(limite5[0][0], limite5[1][0]), abert=0.10, max_it=1000, max_viz=30, pos='view5', salvar=False)
    otimizar(tipo=tipo6, busca='hc', f=funcao6, dominio=limite6, x_otimo=(limite6[0][0], limite6[1][0]), abert=1.40, max_it=1000, max_viz=30, pos='view6', salvar=False)
    otimizar(tipo=tipo7, busca='hc', f=funcao7, dominio=limite7, x_otimo=(limite7[0][0], limite7[1][0]), abert=0.54, max_it=1000, max_viz=30, pos='view7', salvar=False)
    otimizar(tipo=tipo8, busca='hc', f=funcao8, dominio=limite8, x_otimo=(limite8[0][0], limite8[1][0]), abert=0.10, max_it=1000, max_viz=30, pos='view8', salvar=False)

    otimizar(tipo=tipo1, busca='grs', f=funcao1, dominio=limite1, x_otimo=np.random.uniform(low=limite1[0], high=limite1[1], size=(2, )), abert=0, max_it=1000, max_viz=0, pos='view1', salvar=False)
    otimizar(tipo=tipo2, busca='grs', f=funcao2, dominio=limite2, x_otimo=np.random.uniform(low=limite2[0], high=limite2[1], size=(2, )), abert=0, max_it=1000, max_viz=0, pos='view2', salvar=False)
    otimizar(tipo=tipo3, busca='grs', f=funcao3, dominio=limite3, x_otimo=np.random.uniform(low=limite3[0], high=limite3[1], size=(2, )), abert=0, max_it=1000, max_viz=0, pos='view3', salvar=False)
    otimizar(tipo=tipo4, busca='grs', f=funcao4, dominio=limite4, x_otimo=np.random.uniform(low=limite4[0], high=limite4[1], size=(2, )), abert=0, max_it=1000, max_viz=0, pos='view4', salvar=False)
    otimizar(tipo=tipo5, busca='grs', f=funcao5, dominio=limite5, x_otimo=np.random.uniform(low=limite5[0], high=limite5[1], size=(2, )), abert=0, max_it=1000, max_viz=0, pos='view5', salvar=False)
    otimizar(tipo=tipo6, busca='grs', f=funcao6, dominio=limite6, x_otimo=np.random.uniform(low=limite6[0], high=limite6[1], size=(2, )), abert=0, max_it=1000, max_viz=0, pos='view6', salvar=False)
    otimizar(tipo=tipo7, busca='grs', f=funcao7, dominio=limite7, x_otimo=np.random.uniform(low=limite7[0], high=limite7[1], size=(2, )), abert=0, max_it=1000, max_viz=0, pos='view7', salvar=False)
    otimizar(tipo=tipo8, busca='grs', f=funcao8, dominio=limite8, x_otimo=np.random.uniform(low=limite8[0], high=limite8[1], size=(2, )), abert=0, max_it=1000, max_viz=0, pos='view8', salvar=False)
