from PIL import Image
import numpy as num
import pandas as pd
import random as rd
import matplotlib.pyplot as plot
import glob, os

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

def exibir(titulo, funcao, dominio, solucoes, todos):
    # todos = True
    algoritmo, func = titulo.strip().split('\n')
    if algoritmo[0] == 'H':
        hiper = 'e'
    else:
        hiper = 'o'

    if ((funcao.__name__ == 'funcao1') and (algoritmo[0] == 'H')):
        print('Solução ótima')
    if (funcao.__name__ == 'funcao1'):
        print('  -> ' + algoritmo)
    print('     - ' + func)
    print('         '+hiper+'  = {:.4f}'.format(solucoes.at[101, 'hiperparametro']))
    print('         x  = ({:.4f}'.format(solucoes.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(solucoes.at[101, 'pt_otimo[y]'])+')')
    print('       f(x) = {:.4f}'.format(solucoes.at[101, 'f(otimo)']))
    if (funcao.__name__ == 'funcao8'):
        print('')

    # Geração do grid e gráfico da função
    x = num.linspace(start=[dominio[0][0], dominio[1][0]], stop=[dominio[0][1], dominio[1][1]], num=1000, axis=1)
    X1, X2 = num.meshgrid(x[0], x[1])
    Y = funcao(X1, X2)

    # Plotagem do desenho do gráfico e do ponto inicial
    fig = plot.figure()
    ax = fig.add_subplot(projection='3d')
    surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
    ax.scatter(solucoes.at[0, 'pt_otimo[x]'], solucoes.at[0, 'pt_otimo[y]'], solucoes.at[0, 'f(otimo)'], marker='x', s=100, linewidth=1, color='black')
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
            ax.scatter(solucoes.at[i, 'pt_otimo[x]'], solucoes.at[i, 'pt_otimo[y]'], solucoes.at[i, 'f(otimo)'], marker='o', s=20, linewidth=1, color='blue', edgecolors='black')
    # Plotagem do gráfico no melhor ponto ótimo
    ax.scatter(solucoes.at[101, 'pt_otimo[x]'], solucoes.at[101, 'pt_otimo[y]'], solucoes.at[101, 'f(otimo)'], marker='*', s=150, linewidth=1, color='green', edgecolors='black')

    ax.view_init(elev=30, azim=-65, roll=0.)
    plot.show()

def salvar(titulo, funcao, dominio, solucoes, todos):
#  Para todos os algoritmos executados em cada função do problema:
#   -> Salvar arquivo .png com o gráfico estático
#   -> Salvar arquivo .gif com o gráfico animado rotacionando em 360°
#   -> Salvar arquivo .csv com os dados do DataFrame
#   -> Salvar arquivo .txt com a solução apresentada
    # todos = True
    nome_arquivo = funcao.__name__

    algoritmo, _ = titulo.strip().split('\n')
    if algoritmo[0] == 'H':
        nome_arquivo += '.1-hc'
        hiper = 'e'
    elif algoritmo[0] == 'L':
        nome_arquivo += '.2-lrs'
        hiper = 'o'
    elif algoritmo[0] == 'G':
        nome_arquivo += '.3-grs'
        hiper = 'o'

    x = num.linspace(start=[dominio[0][0], dominio[1][0]], stop=[dominio[0][1], dominio[1][1]], num=1000, axis=1)
    X1, X2 = num.meshgrid(x[0], x[1])
    Y = funcao(X1, X2)

    fig = plot.figure()
    ax = fig.add_subplot(projection='3d')
    surface = ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='jet')
    ax.scatter(solucoes.at[0, 'pt_otimo[x]'], solucoes.at[0, 'pt_otimo[y]'], solucoes.at[0, 'f(otimo)'], marker='x', s=100, linewidth=1, color='black')

    ax.set_title(titulo)
    ax.set_xlabel('valores x')
    ax.set_ylabel('valores y')
    ax.set_zlabel('valores z')
    plot.tight_layout()

    if todos:
        for i in range(99):
            i += 1
            ax.scatter(solucoes.at[i, 'pt_otimo[x]'], solucoes.at[i, 'pt_otimo[y]'], solucoes.at[i, 'f(otimo)'], marker='o', s=20, linewidth=1, color='blue', edgecolors='black')
    ax.scatter(solucoes.at[101, 'pt_otimo[x]'], solucoes.at[101, 'pt_otimo[y]'], solucoes.at[101, 'f(otimo)'], marker='*', s=150, linewidth=1, color='green', edgecolors='black')

    ax.view_init(elev=10, azim=-65, roll=0.)

    # verifica e cria os diretorios necessarios para salvar os arquivos
    if not os.path.exists('problema1'):
        os.makedirs('problema1')
    if not os.path.exists('problema1/dados'):
        os.makedirs('problema1/dados')
    if not os.path.exists('problema1/imagens'):
        os.makedirs('problema1/imagens')
    if not os.path.exists('problema1/imagens/temp'):
        os.makedirs('problema1/imagens/temp')
    # cria a imagem .png estática do gráfico
    plot.savefig('problema1/imagens/%s.png' % nome_arquivo)
    # cria-se 360 imagens .png com o movimento de rotação de 360° no eixo Z
    compress = 3  # compressão do gif animado (variando de 1 a 6), quanto maior o número menos frames o gif vai ter e menor o espaço em disco que vai ocupar (melhor definição é 1, aconselho deixar em 3 que tem o melhor benefício)
    for ii in range(0, 360, compress):
        ax.view_init(elev=10., azim=-ii, roll=0.)
        plot.savefig('problema1/imagens/temp/movie%0*d.png' % (3, ii))
    fp_in = 'problema1/imagens/temp/movie*.png'
    fp_out = 'problema1/imagens/%s.gif' % nome_arquivo
    # carrega todos esses pngs
    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    # salva como arquivo gif animado
    img.save(fp=fp_out, format='GIF', append_images=imgs, save_all=True, duration=100, loop=0) # loop=0 (define o loop infinito)
    # salva o DataFrame como arquivo .csv
    solucoes.to_csv('problema1/dados/%s.csv' % nome_arquivo, index=False)
    # salva a solução da função como um arquivo de texto .txt
    texto = 'Solução ótima:\n'
    texto += '  -> ' + funcao.__name__ + '\n'
    texto += '     - ' + algoritmo + '\n'
    texto += '         '+hiper+'  = {:.4f}'.format(solucoes.at[101, 'hiperparametro']) + '\n'
    texto += '         x  = ({:.4f}'.format(solucoes.at[101, 'pt_otimo[x]'])+', {:.4f}'.format(solucoes.at[101, 'pt_otimo[y]'])+')' + '\n'
    texto += '       f(x) = {:.4f}'.format(solucoes.at[101, 'f(otimo)'])
    with open('problema1/dados/%s.txt' % nome_arquivo, 'w') as arquivo:
        arquivo.write(texto)

def perturb_hc(x, hiper, limites):
    res = num.random.uniform(low=x[0]-hiper, high=x[1]+hiper, size=(2, )) # |x − y| ≤ ε
    # verificação da restrição dos limites da função
    if res[0] < limites[0][0]:
        res[0] = limites[0][0]
    elif res[0] > limites[0][1]:
        res[0] = limites[0][1]
    if res[1] < limites[1][0]:
        res[1] = limites[1][0]
    elif res[1] > limites[1][1]:
        res[1] = limites[1][1]
    return res

def perturb_lrs(x, hiper, limites):
    n = num.random.normal(loc=0, scale=hiper, size=(2,)) # distribuicao normal: n ~ N(0, σ)
    x_normal = num.add(x, n) # x + n
    res = num.random.uniform(low=x_normal[0], high=x_normal[1], size=(2, )) # distribuição uniforme: U(x[0], x[1])
    # verificação da restrição dos limites da função
    if res[0] < limites[0][0]:
        res[0] = limites[0][0]
    elif res[0] > limites[0][1]:
        res[0] = limites[0][1]
    if res[1] < limites[1][0]:
        res[1] = limites[1][0]
    elif res[1] > limites[1][1]:
        res[1] = limites[1][1]
    return res

# def perturb_grs(x, hiper, limites):
    # res = num.random.uniform(low=x[0]*hiper, high=x[1]*hiper, size=(2, ))
def perturb_grs(limites):
    res = num.random.uniform(low=(limites[0][0], limites[1][0]), high=(limites[0][1], limites[1][1]), size=(2, )) 
    # verificação da restrição dos limites da função
    if res[0] < limites[0][0]:
        res[0] = limites[0][0]
    elif res[0] > limites[0][1]:
        res[0] = limites[0][1]
    if res[1] < limites[1][0]:
        res[1] = limites[1][0]
    elif res[1] > limites[1][1]:
        res[1] = limites[1][1]
    return res

def otimizar(tipo, busca, f, dominio, max_iteracoes, max_vizinhos):
    # Criação do DataFrame vazio que armazenará as soluções de cada rodada
    solucoes = pd.DataFrame(columns = ['pt_otimo[x]', 'pt_otimo[y]', 'f(otimo)', 'hiperparametro'])

    # Geração do ponto inicial
    if (busca == 'hc'):
        x_inicial = (dominio[0][0], dominio[1][0])
    else: # para os algoritmos de busca 'lrs' e 'grs'
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

    for z in range(100): # Executa uma sequência de 100 rodadas da busca
        # Algoritmo de busca: Hill Climbing
        if (busca == 'hc'):
            hiperparametro = 0.1 # valor imposto pelo problema
            i = 0 # contador de iterações
            melhoria = True # define melhoria
            while i < max_iteracoes and melhoria: # enquanto náo ocorrer todas as iterações e houver melhoria
                melhoria = False # considera que não houve melhoria até ser testado
                for _ in range(max_vizinhos): # enquanto náo percorrer toda a vizinhança
                    x_cand = perturb_hc(x_otimo, hiperparametro, dominio) # geração do novo candidato
                    f_cand = f(x_cand[0], x_cand[1]) # avalia candidato aplicando na função
                    if teste(f_cand, f_otimo): # validação do candidato
                        melhoria = True # assume que houve melhoria
                        x_otimo = x_cand # assume o valor do ótimo
                        f_otimo = f_cand # assume o valor da função do ótimo
                        break # encerra o while das iterações, porque encontrou um novo ótimo
                i += 1 # incremento do contador de iterações
        # Algoritmo de busca: Local Random Search
        elif (busca == 'lrs'):
            hiperparametro = num.random.uniform(0, 1) # desvio padrão: número aleatório entre 0 e 1 (0 < sigma < 1)
            i = 0 # contador de iterações
            while i < max_iteracoes: # enquanto náo ocorrer todas as iterações
                x_cand = perturb_lrs(x_otimo, hiperparametro, dominio) # geração do novo candidato
                f_cand = f(x_cand[0], x_cand[1]) # avalia candidato aplicando na função
                if teste(f_cand, f_otimo): # validação do candidato
                    x_otimo = x_cand # assume o valor do ótimo
                    f_otimo = f_cand # assume o valor da função do ótimo
                    break # encerra o while das iterações, porque encontrou um novo ótimo
                i += 1 # incremento do contador de iterações
        # Algoritmo de busca: Global Random Search
        elif (busca == 'grs'):
            hiperparametro = 0 # não interessa para esse algoritmo
            i = 0 # contador de iterações
            while i < max_iteracoes: # enquanto náo ocorrer todas as iterações
                x_cand = perturb_grs(dominio) # geração do novo candidato
                f_cand = f(x_cand[0], x_cand[1]) # avalia candidato aplicando na função
                if teste(f_cand, f_otimo): # validação do candidato
                    x_otimo = x_cand # assume o valor do ótimo
                    f_otimo = f_cand # assume o valor da função do ótimo
                    break # encerra o while das iterações, porque encontrou um novo ótimo
                i += 1 # incremento do contador de iterações

        # Adição da solução encontrada nesta rodada no DataFrame de soluções
        solucoes.loc[z+1] = [x_otimo[0], x_otimo[1], f_otimo, hiperparametro]

        # Testa se o ótimo dessa rodado foi melhor do que o da rodada passada
        if teste(f_otimo, f_melhor):
            x_melhor = x_otimo
            f_melhor = f_otimo

        # Adiciona no DataFrame a melhor solução encontrada de todas as rodadas
        if (z == 99):
            solucoes.loc[z+2] = [x_melhor[0], x_melhor[1], f_melhor, hiperparametro]
    return solucoes

try:
    if __name__ == '__main__':
        # df = otimizar(tipo=tipo1, busca='hc', f=funcao1, dominio=limite1, max_iteracoes=1000, max_vizinhos=30)
        # exibir(titulo='Hill Climbing\nFunção 1 ('+tipo1+')', funcao=funcao1, dominio=limite1, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo2, busca='hc', f=funcao2, dominio=limite2, max_iteracoes=1000, max_vizinhos=30)
        # exibir(titulo='Hill Climbing\nFunção 2 ('+tipo2+')', funcao=funcao2, dominio=limite2, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo3, busca='hc', f=funcao3, dominio=limite3, max_iteracoes=1000, max_vizinhos=30)
        # exibir(titulo='Hill Climbing\nFunção 3 ('+tipo3+')', funcao=funcao3, dominio=limite3, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo4, busca='hc', f=funcao4, dominio=limite4, max_iteracoes=1000, max_vizinhos=30)
        # exibir(titulo='Hill Climbing\nFunção 4 ('+tipo4+')', funcao=funcao4, dominio=limite4, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo5, busca='hc', f=funcao5, dominio=limite5, max_iteracoes=1000, max_vizinhos=30)
        # exibir(titulo='Hill Climbing\nFunção 5 ('+tipo5+')', funcao=funcao5, dominio=limite5, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo6, busca='hc', f=funcao6, dominio=limite6, max_iteracoes=1000, max_vizinhos=30)
        # exibir(titulo='Hill Climbing\nFunção 6 ('+tipo6+')', funcao=funcao6, dominio=limite6, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo7, busca='hc', f=funcao7, dominio=limite7, max_iteracoes=1000, max_vizinhos=30)
        # exibir(titulo='Hill Climbing\nFunção 7 ('+tipo7+')', funcao=funcao7, dominio=limite7, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo8, busca='hc', f=funcao8, dominio=limite8, max_iteracoes=1000, max_vizinhos=30)
        # exibir(titulo='Hill Climbing\nFunção 8 ('+tipo8+')', funcao=funcao8, dominio=limite8, solucoes=df, todos=False)

        # df = otimizar(tipo=tipo1, busca='lrs', f=funcao1, dominio=limite1, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Local Random Search\nFunção 1 ('+tipo1+')', funcao=funcao1, dominio=limite1, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo2, busca='lrs', f=funcao2, dominio=limite2, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Local Random Search\nFunção 2 ('+tipo2+')', funcao=funcao2, dominio=limite2, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo3, busca='lrs', f=funcao3, dominio=limite3, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Local Random Search\nFunção 3 ('+tipo3+')', funcao=funcao3, dominio=limite3, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo4, busca='lrs', f=funcao4, dominio=limite4, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Local Random Search\nFunção 4 ('+tipo4+')', funcao=funcao4, dominio=limite4, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo5, busca='lrs', f=funcao5, dominio=limite5, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Local Random Search\nFunção 5 ('+tipo5+')', funcao=funcao5, dominio=limite5, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo6, busca='lrs', f=funcao6, dominio=limite6, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Local Random Search\nFunção 6 ('+tipo6+')', funcao=funcao6, dominio=limite6, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo7, busca='lrs', f=funcao7, dominio=limite7, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Local Random Search\nFunção 7 ('+tipo7+')', funcao=funcao7, dominio=limite7, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo8, busca='lrs', f=funcao8, dominio=limite8, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Local Random Search\nFunção 8 ('+tipo8+')', funcao=funcao8, dominio=limite8, solucoes=df, todos=False)

        # df = otimizar(tipo=tipo1, busca='grs', f=funcao1, dominio=limite1, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Global Random Search\nFunção 1 ('+tipo1+')', funcao=funcao1, dominio=limite1, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo2, busca='grs', f=funcao2, dominio=limite2, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Global Random Search\nFunção 2 ('+tipo2+')', funcao=funcao2, dominio=limite2, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo3, busca='grs', f=funcao3, dominio=limite3, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Global Random Search\nFunção 3 ('+tipo3+')', funcao=funcao3, dominio=limite3, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo4, busca='grs', f=funcao4, dominio=limite4, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Global Random Search\nFunção 4 ('+tipo4+')', funcao=funcao4, dominio=limite4, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo5, busca='grs', f=funcao5, dominio=limite5, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Global Random Search\nFunção 5 ('+tipo5+')', funcao=funcao5, dominio=limite5, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo6, busca='grs', f=funcao6, dominio=limite6, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Global Random Search\nFunção 6 ('+tipo6+')', funcao=funcao6, dominio=limite6, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo7, busca='grs', f=funcao7, dominio=limite7, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Global Random Search\nFunção 7 ('+tipo7+')', funcao=funcao7, dominio=limite7, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo8, busca='grs', f=funcao8, dominio=limite8, max_iteracoes=1000, max_vizinhos=0)
        # exibir(titulo='Global Random Search\nFunção 8 ('+tipo8+')', funcao=funcao8, dominio=limite8, solucoes=df, todos=False)


        # # utilizar essa funcao de salvar com moderação, pois para executar os três algoritmos nas oito funções o projeto levou 3695.844 segundos para salvar todos os arquivos
        df = otimizar(tipo=tipo1, busca='hc', f=funcao1, dominio=limite1, max_iteracoes=1000, max_vizinhos=30)
        salvar(titulo='Hill Climbing\nFunção 1 ('+tipo1+')', funcao=funcao1, dominio=limite1, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo2, busca='hc', f=funcao2, dominio=limite2, max_iteracoes=1000, max_vizinhos=30)
        # salvar(titulo='Hill Climbing\nFunção 2 ('+tipo2+')', funcao=funcao2, dominio=limite2, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo3, busca='hc', f=funcao3, dominio=limite3, max_iteracoes=1000, max_vizinhos=30)
        # salvar(titulo='Hill Climbing\nFunção 3 ('+tipo3+')', funcao=funcao3, dominio=limite3, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo4, busca='hc', f=funcao4, dominio=limite4, max_iteracoes=1000, max_vizinhos=30)
        # salvar(titulo='Hill Climbing\nFunção 4 ('+tipo4+')', funcao=funcao4, dominio=limite4, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo5, busca='hc', f=funcao5, dominio=limite5, max_iteracoes=1000, max_vizinhos=30)
        # salvar(titulo='Hill Climbing\nFunção 5 ('+tipo5+')', funcao=funcao5, dominio=limite5, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo6, busca='hc', f=funcao6, dominio=limite6, max_iteracoes=1000, max_vizinhos=30)
        # salvar(titulo='Hill Climbing\nFunção 6 ('+tipo6+')', funcao=funcao6, dominio=limite6, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo7, busca='hc', f=funcao7, dominio=limite7, max_iteracoes=1000, max_vizinhos=30)
        # salvar(titulo='Hill Climbing\nFunção 7 ('+tipo7+')', funcao=funcao7, dominio=limite7, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo8, busca='hc', f=funcao8, dominio=limite8, max_iteracoes=1000, max_vizinhos=30)
        # salvar(titulo='Hill Climbing\nFunção 8 ('+tipo8+')', funcao=funcao8, dominio=limite8, solucoes=df, todos=False)

        # df = otimizar(tipo=tipo1, busca='lrs', f=funcao1, dominio=limite1, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Local Random Search\nFunção 1 ('+tipo1+')', funcao=funcao1, dominio=limite1, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo2, busca='lrs', f=funcao2, dominio=limite2, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Local Random Search\nFunção 2 ('+tipo2+')', funcao=funcao2, dominio=limite2, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo3, busca='lrs', f=funcao3, dominio=limite3, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Local Random Search\nFunção 3 ('+tipo3+')', funcao=funcao3, dominio=limite3, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo4, busca='lrs', f=funcao4, dominio=limite4, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Local Random Search\nFunção 4 ('+tipo4+')', funcao=funcao4, dominio=limite4, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo5, busca='lrs', f=funcao5, dominio=limite5, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Local Random Search\nFunção 5 ('+tipo5+')', funcao=funcao5, dominio=limite5, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo6, busca='lrs', f=funcao6, dominio=limite6, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Local Random Search\nFunção 6 ('+tipo6+')', funcao=funcao6, dominio=limite6, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo7, busca='lrs', f=funcao7, dominio=limite7, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Local Random Search\nFunção 7 ('+tipo7+')', funcao=funcao7, dominio=limite7, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo8, busca='lrs', f=funcao8, dominio=limite8, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Local Random Search\nFunção 8 ('+tipo8+')', funcao=funcao8, dominio=limite8, solucoes=df, todos=False)

        # df = otimizar(tipo=tipo1, busca='grs', f=funcao1, dominio=limite1, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Global Random Search\nFunção 1 ('+tipo1+')', funcao=funcao1, dominio=limite1, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo2, busca='grs', f=funcao2, dominio=limite2, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Global Random Search\nFunção 2 ('+tipo2+')', funcao=funcao2, dominio=limite2, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo3, busca='grs', f=funcao3, dominio=limite3, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Global Random Search\nFunção 3 ('+tipo3+')', funcao=funcao3, dominio=limite3, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo4, busca='grs', f=funcao4, dominio=limite4, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Global Random Search\nFunção 4 ('+tipo4+')', funcao=funcao4, dominio=limite4, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo5, busca='grs', f=funcao5, dominio=limite5, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Global Random Search\nFunção 5 ('+tipo5+')', funcao=funcao5, dominio=limite5, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo6, busca='grs', f=funcao6, dominio=limite6, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Global Random Search\nFunção 6 ('+tipo6+')', funcao=funcao6, dominio=limite6, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo7, busca='grs', f=funcao7, dominio=limite7, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Global Random Search\nFunção 7 ('+tipo7+')', funcao=funcao7, dominio=limite7, solucoes=df, todos=False)
        # df = otimizar(tipo=tipo8, busca='grs', f=funcao8, dominio=limite8, max_iteracoes=1000, max_vizinhos=0)
        # salvar(titulo='Global Random Search\nFunção 8 ('+tipo8+')', funcao=funcao8, dominio=limite8, solucoes=df, todos=False)
finally:
    # exclui os arquivos temporários gerados
    if os.path.exists('problema1/imagens/temp'):
        os.remove('problema1/imagens/temp')
