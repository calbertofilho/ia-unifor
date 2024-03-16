import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot

# Problema 1 (Minimizar)
tipo1 = 'min'
limite1 = [(-100, 100), (-100, 100)]
def funcao1(x1, x2):
    return (x1 ** 2 + x2 ** 2)
# Problema 2 (Maximizar)
tipo2 = 'max'
limite2 = [(-2, 4), (-2, 5)]
def funcao2(x1, x2):
    return (num.exp(-(x1 ** 2 + x2 ** 2)) + 2 * num.exp(-((x1 - 1.7) ** 2 + (x2 - 1.7) ** 2)))
# Problema 3 (Minimizar)
tipo3 = 'min'
limite3 = [(-8, 8), (-8, 8)]
def funcao3(x1, x2):
    return (-20 * num.exp(-0.2 * num.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))) - num.exp(0.5 * (num.cos(2 * num.pi * x1) + num.cos(2 * num.pi * x2))) + 20 + num.exp(1)
# Problema 4 (Minimizar)
tipo4 = 'min'
limite4 = [(-5.12, 5.12), (-5.12, 5.12)]
def funcao4(x1, x2):
    return ((x1 ** 2 - 10 * num.cos(2 * num.pi * x1) + 10) + (x2 ** 2 - 10 * num.cos(2 * num.pi * x2) + 10))
# Problema 5 (Maximizar)
tipo5 = 'max'
limite5 = [(-10, 10), (-10, 10)]
def funcao5(x1, x2):
    return (((x1 * num.cos(x1)) / 20) + 2 * num.exp(-(x1 ** 2) - ((x2 - 1) ** 2)) + 0.01 * x1 * x2)
# Problema 6 (Maximizar)
tipo6 = 'max'
limite6 = [(-1, 3), (-1, 3)]
def funcao6(x1, x2):
    return ((x1 * num.sin(4 * num.pi * x1)) - (x2 * num.sin((4 * num.pi * x2) + num.pi)) + 1)
# Problema 7 (Minimizar)
tipo7 = 'min'
limite7 = [(0, num.pi), (0, num.pi)]
def funcao7(x1, x2):
    return ((-num.sin(x1) * num.sin((x1 ** 2)/num.pi) ** (2 * 10)) - (num.sin(x2) * (num.sin((2 * x2 ** 2)/num.pi) ** (2 * 10))))
# Problema 8 (Minimizar)
tipo8 = 'min'
limite8 = [(-200, 20), (-200, 20)]
def funcao8(x1, x2):
    return ((-(x2 + 47)) * num.sin(num.sqrt(num.abs((x1 / 2) + (x2 + 47))))) - (x1 * num.sin(num.sqrt(num.abs(x1 - (x2 + 47)))))

def desenhar(titulo, funcao, dominio, solucoes, todos):
    # Geração do grid e gráfico da função
    x = num.linspace(start=[dominio[0][0], dominio[1][0]], stop=[dominio[0][1], dominio[1][1]], num=1000, axis=1)
    X1, X2 = num.meshgrid(x[0], x[1])
    Y = funcao(X1, X2)

    # Plotagem do desenho do gráfico e do ponto inicial
    fig = plot.figure()
    ax = fig.add_subplot(projection='3d')
    surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
    ax.scatter(solucoes.at[0, 'pt_otimo[x]'], solucoes.at[0, 'pt_otimo[y]'], solucoes.at[0, 'f(otimo)'], marker='x', s=80, linewidth=1, color='red', alpha=0.6)
    # plot.colorbar(surface)

    # Etiquetas dos eixos
    ax.set_title(titulo)
    ax.set_xlabel('valores x')
    ax.set_ylabel('valores y')
    ax.set_zlabel('valores z')
    plot.tight_layout()  # Melhora o ajuste para a imagem a ser plotada

    if todos:
        # Plotagem de todos os pontos ótimos das 100 rodadas no gráfico
        for i in range(99):
            i += 1
            ax.scatter(solucoes.at[i, 'pt_otimo[x]'], solucoes.at[i, 'pt_otimo[y]'], solucoes.at[i, 'f(otimo)'], marker='o', s=20, linewidth=1, color='blue', edgecolors='black', alpha=0.6)
    # Plotagem do gráfico no melhor ponto ótimo
    ax.scatter(solucoes.at[101, 'pt_otimo[x]'], solucoes.at[101, 'pt_otimo[y]'], solucoes.at[101, 'f(otimo)'], marker='*', s=150, linewidth=1, color='green', edgecolors='black', alpha=0.6)

    ax.view_init(elev=30, azim=-65, roll=0.)
    plot.show()

def salvar():
#    # Posicionamento da camera de visualização 3d
#    # ax.view_init(pos)
#    if (pos == 'view1'):
#        ax.view_init(elev=10., azim=-65., roll=0.)
#    elif (pos == 'view2'):
#        ax.view_init(elev=24., azim=-66., roll=0.)
#    elif (pos == 'view3'):
#        ax.view_init(elev=30., azim=-65., roll=0.)
#    elif (pos == 'view4'):
#        #ax.view_init(elev=25., azim=-61., roll=0.)
#        ax.view_init(elev=6., azim=-62., roll=0.)
#    elif (pos == 'view5'):
#        #ax.view_init(elev=15., azim=-140., roll=0.)
#        ax.view_init(elev=35., azim=-91., roll=0.)
#    elif (pos == 'view6'):
#        ax.view_init(elev=30., azim=-60., roll=0.)
#    elif (pos == 'view7'):
#        ax.view_init(elev=26., azim=-65., roll=0.)
#    elif (pos == 'view8'):
#        #ax.view_init(elev=30., azim=160., roll=0.)
#        ax.view_init(elev=18., azim=-125., roll=0.)
#
#    # Gerando os arquivos com as soluções encontrados (Imagem (PNG) do gráfico com apresentação do ponto inicial e o ponto ótimo e do DataFrame (CSV) de soluções da sequência de 100 rodadas)
#    #if salvar:
#    #    nome_arquivo = 'av1-'+busca+'_'+f.__name__
#    #    solucoes.to_csv(nome_arquivo+'.csv', index=False)
#    #    plot.savefig(nome_arquivo+'.png')
    ...

def perturb_hc(x, abertura, limites):
    res = num.random.uniform(low=x[0]-abertura, high=x[1]+abertura, size=(2, ))
    if res[0] < limites[0][0]:
        res[0] = limites[0][0]
    elif res[0] > limites[0][1]:
        res[0] = limites[0][1]
    if res[1] < limites[1][0]:
        res[1] = limites[1][0]
    elif res[1] > limites[1][1]:
        res[1] = limites[1][1]
    return res

def perturb_rs(x, abertura, limites):
    res = num.random.uniform(low=x[0]*abertura, high=x[1]*abertura, size=(2, ))
    if res[0] < limites[0][0]:
        res[0] = limites[0][0]
    elif res[0] > limites[0][1]:
        res[0] = limites[0][1]
    if res[1] < limites[1][0]:
        res[1] = limites[1][0]
    elif res[1] > limites[1][1]:
        res[1] = limites[1][1]
    return res

def otimizar(tipo, busca, f, dominio, max_it, max_viz):
    # Criação do DataFrame vazio que armazenará as soluções de cada rodada
    solucoes = pd.DataFrame(columns = ['pt_otimo[x]', 'pt_otimo[y]', 'f(otimo)', 'abertura'])

    # Geração do ponto inicial
    if (busca == 'hc'):
        x_inicial = (dominio[0][0], dominio[1][0])
    elif (busca == 'lrs'):
        x_inicial = (0, 0)
    elif (busca == 'grs'):
        x_inicial = num.random.uniform(low=(dominio[0][0], dominio[1][0]), high=(dominio[0][1], dominio[1][1]), size=(2, ))
    x_melhor = x_otimo = x_inicial
    f_melhor = f_otimo = f(x_otimo[0], x_otimo[1])

    # Adição do ponto inicial no DataFrame de soluções
    solucoes.loc[0] = [x_otimo[0], x_otimo[1], f_otimo, 0]

    # Define o tipo de teste que deve ser feito, para maximiar ou minimizar a função
    def teste(candidato, atual):
        if (tipo == 'min'): # Minimizar
            return (candidato < atual)
        elif (tipo == 'max'): # Maximizar
            return (candidato > atual)

    for z in range(100):
        # Algoritmo da busca por Hill Climbing
        if (busca == 'hc'):
            abert = rd.random()
            max_iter = max_it
            max_vizinhos = max_viz
            i = 0
            melhoria = True

            while i < max_iter and melhoria:
                melhoria = False
                for _ in range(max_vizinhos):
                    x_cand = perturb_hc(x_otimo, abert, dominio)
                    f_cand = f(x_cand[0], x_cand[1])
                    if teste(f_cand, f_otimo):
                        melhoria = True
                        x_otimo = x_cand
                        f_otimo = f_cand
                        break
                i += 1
        # Algoritmo da busca por Local Random Search
        elif (busca == 'lrs'): #https://freedium.cfd/https://medium.com/analytics-vidhya/how-does-random-search-algorithm-work-python-implementation-b69e779656d6
            ... 
        # Algoritmo da busca por Global Random Search
        elif (busca == 'grs'):
            abert = rd.random()
            max = max_it
            i = 0

            while i < max:
                x_cand = perturb_rs(x_otimo, abert, dominio)
                f_cand = f(x_cand[0], x_cand[1])
                if teste(f_cand, f_otimo):
                    x_otimo = x_cand
                    f_otimo = f_cand
                    break
                i += 1

        # Adição da solução encontrada nesta rodada no DataFrame de soluções
        solucoes.loc[z+1] = [x_otimo[0], x_otimo[1], f_otimo, abert]

        # Testa se o ótimo dessa rodado foi melhor do que o da rodada passada
        if teste(f_otimo, f_melhor):
            x_melhor = x_otimo
            f_melhor = f_otimo

        # Adiciona no DataFrame a melhor solução encontrada de todas as rodadas
        if (z == 99):
            solucoes.loc[z+2] = [x_melhor[0], x_melhor[1], f_melhor, abert]

    return solucoes

if __name__ == '__main__':
    print('Solução ótima')
    print('  -> Hill Climbing')
    print('     - Função 1 ('+tipo1+')')
    df = otimizar(tipo=tipo1, busca='hc', f=funcao1, dominio=limite1, max_it=1000, max_viz=30)
    print('         ε  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Hill Climbing\nFunção 1 ('+tipo1+')', funcao=funcao1, dominio=limite1, solucoes=df, todos=True)
    print('     - Função 2 ('+tipo2+')')
    df = otimizar(tipo=tipo2, busca='hc', f=funcao2, dominio=limite2, max_it=1000, max_viz=30)
    print('         ε  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Hill Climbing\nFunção 2 ('+tipo2+')', funcao=funcao2, dominio=limite2, solucoes=df, todos=True)
    print('     - Função 3 ('+tipo3+')')
    df = otimizar(tipo=tipo3, busca='hc', f=funcao3, dominio=limite3, max_it=1000, max_viz=30)
    print('         ε  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Hill Climbing\nFunção 3 ('+tipo3+')', funcao=funcao3, dominio=limite3, solucoes=df, todos=True)
    print('     - Função 4 ('+tipo4+')')
    df = otimizar(tipo=tipo4, busca='hc', f=funcao4, dominio=limite4, max_it=1000, max_viz=30)
    print('         ε  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Hill Climbing\nFunção 4 ('+tipo4+')', funcao=funcao4, dominio=limite4, solucoes=df, todos=True)
    print('     - Função 5 ('+tipo5+')')
    df = otimizar(tipo=tipo5, busca='hc', f=funcao5, dominio=limite5, max_it=1000, max_viz=30)
    print('         ε  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Hill Climbing\nFunção 5 ('+tipo5+')', funcao=funcao5, dominio=limite5, solucoes=df, todos=True)
    print('     - Função 6 ('+tipo6+')')
    df = otimizar(tipo=tipo6, busca='hc', f=funcao6, dominio=limite6, max_it=1000, max_viz=30)
    print('         ε  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Hill Climbing\nFunção 6 ('+tipo6+')', funcao=funcao6, dominio=limite6, solucoes=df, todos=True)
    print('     - Função 7 ('+tipo7+')')
    df = otimizar(tipo=tipo7, busca='hc', f=funcao7, dominio=limite7, max_it=1000, max_viz=30)
    print('         ε  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Hill Climbing\nFunção 7 ('+tipo7+')', funcao=funcao7, dominio=limite7, solucoes=df, todos=True)
    print('     - Função 8 ('+tipo8+')')
    df = otimizar(tipo=tipo8, busca='hc', f=funcao8, dominio=limite8, max_it=1000, max_viz=30)
    print('         ε  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Hill Climbing\nFunção 8 ('+tipo8+')', funcao=funcao8, dominio=limite8, solucoes=df, todos=True)
    print('')
    print('  -> Local Random Search')
    print('     - Função 1 ('+tipo1+')')
    df = otimizar(tipo=tipo1, busca='lrs', f=funcao1, dominio=limite1, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Local Random Search\nFunção 1 ('+tipo1+')', funcao=funcao1, dominio=limite1, solucoes=df, todos=True)
    print('     - Função 2 ('+tipo2+')')
    df = otimizar(tipo=tipo2, busca='lrs', f=funcao2, dominio=limite2, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Local Random Search\nFunção 2 ('+tipo2+')', funcao=funcao2, dominio=limite2, solucoes=df, todos=True)
    print('     - Função 3 ('+tipo3+')')
    df = otimizar(tipo=tipo3, busca='lrs', f=funcao3, dominio=limite3, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Local Random Search\nFunção 3 ('+tipo3+')', funcao=funcao3, dominio=limite3, solucoes=df, todos=True)
    print('     - Função 4 ('+tipo4+')')
    df = otimizar(tipo=tipo4, busca='lrs', f=funcao4, dominio=limite4, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Local Random Search\nFunção 4 ('+tipo4+')', funcao=funcao4, dominio=limite4, solucoes=df, todos=True)
    print('     - Função 5 ('+tipo5+')')
    df = otimizar(tipo=tipo5, busca='lrs', f=funcao5, dominio=limite5, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Local Random Search\nFunção 5 ('+tipo5+')', funcao=funcao5, dominio=limite5, solucoes=df, todos=True)
    print('     - Função 6 ('+tipo6+')')
    df = otimizar(tipo=tipo6, busca='lrs', f=funcao6, dominio=limite6, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Local Random Search\nFunção 6 ('+tipo6+')', funcao=funcao6, dominio=limite6, solucoes=df, todos=True)
    print('     - Função 7 ('+tipo7+')')
    df = otimizar(tipo=tipo7, busca='lrs', f=funcao7, dominio=limite7, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Local Random Search\nFunção 7 ('+tipo7+')', funcao=funcao7, dominio=limite7, solucoes=df, todos=True)
    print('     - Função 8 ('+tipo8+')')
    df = otimizar(tipo=tipo8, busca='lrs', f=funcao8, dominio=limite8, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Local Random Search\nFunção 8 ('+tipo8+')', funcao=funcao8, dominio=limite8, solucoes=df, todos=True)
    print('')
    print('  -> Global Random Search')
    print('     - Função 1 ('+tipo1+')')
    df = otimizar(tipo=tipo1, busca='grs', f=funcao1, dominio=limite1, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Global Random Search\nFunção 1 ('+tipo1+')', funcao=funcao1, dominio=limite1, solucoes=df, todos=True)
    print('     - Função 2 ('+tipo2+')')
    df = otimizar(tipo=tipo2, busca='grs', f=funcao2, dominio=limite2, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Global Random Search\nFunção 2 ('+tipo2+')', funcao=funcao2, dominio=limite2, solucoes=df, todos=True)
    print('     - Função 3 ('+tipo3+')')
    df = otimizar(tipo=tipo3, busca='grs', f=funcao3, dominio=limite3, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Global Random Search\nFunção 3 ('+tipo3+')', funcao=funcao3, dominio=limite3, solucoes=df, todos=True)
    print('     - Função 4 ('+tipo4+')')
    df = otimizar(tipo=tipo4, busca='grs', f=funcao4, dominio=limite4, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Global Random Search\nFunção 4 ('+tipo4+')', funcao=funcao4, dominio=limite4, solucoes=df, todos=True)
    print('     - Função 5 ('+tipo5+')')
    df = otimizar(tipo=tipo5, busca='grs', f=funcao5, dominio=limite5, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Global Random Search\nFunção 5 ('+tipo5+')', funcao=funcao5, dominio=limite5, solucoes=df, todos=True)
    print('     - Função 6 ('+tipo6+')')
    df = otimizar(tipo=tipo6, busca='grs', f=funcao6, dominio=limite6, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Global Random Search\nFunção 6 ('+tipo6+')', funcao=funcao6, dominio=limite6, solucoes=df, todos=True)
    print('     - Função 7 ('+tipo7+')')
    df = otimizar(tipo=tipo7, busca='grs', f=funcao7, dominio=limite7, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Global Random Search\nFunção 7 ('+tipo7+')', funcao=funcao7, dominio=limite7, solucoes=df, todos=True)
    print('     - Função 8 ('+tipo8+')')
    df = otimizar(tipo=tipo8, busca='grs', f=funcao8, dominio=limite8, max_it=1000, max_viz=0)
    print('         σ  = {:.4f}'.format(df.at[101, 'abertura']))
    print('         x  = ({:.4f}'.format(df.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(df.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(df.at[101, 'f(otimo)']))
    desenhar(titulo='Global Random Search\nFunção 8 ('+tipo8+')', funcao=funcao8, dominio=limite8, solucoes=df, todos=True)
