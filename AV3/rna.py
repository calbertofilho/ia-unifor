import os
import sys
import numpy as num
import pandas as pd
import matplotlib.pyplot as plot
from classificadores.mlp import MultilayerPerceptron
from classificadores.adaline import Adaline
from classificadores.perceptron import Perceptron
from utils.manipulation import clearScreen, loadData, shuffleData, partitionData
from utils.progress import printProgressBar, printAnimatedBar

# Referências:
#  - redes-neurais-artificiais-ivan-nunes-da-silva-2ed-2016.pdf
#  - Sistemas_Inteligentes_RNA__Atualiza_Template_.pdf
#  - https://sebastianraschka.com/faq/docs/diff-perceptron-adaline-neuralnet.html
#  - https://juliocprocha.wordpress.com/2020/03/30/introducao-ao-perceptron-multicamadas-passo-a-passo/

def run(inputData: pd.DataFrame, algoritmo: object) -> None:
    rodada = 0                                                                  # Contador das rodadas
    rodadas = 100                                                               # Número máximo de rodadas
    dados_rodada = []                                                           # Coleta de dados de cada rodada
    calculos = []                                                               # Armazena os cálculos
    classificador = algoritmo                                                   # Algoritmo que vai executar a classificação
    nomeClassificador = algoritmo.__class__.__name__                            # Nome do classificador
    while (rodada < rodadas):
        printProgressBar((rodada / rodadas) * 100, 'Calculando...')
        data = shuffleData(inputData)                                           # Embaralha os dados
        X_trn, X_tst, y_trn, y_tst = partitionData(data, 0.8)                   # Particiona os dados no percentual proposto (80% - 20%)
        classificador.treinamento(X_trn, y_trn)                                 # Treina o classificador com os dados separados para treinamento
        y = num.array(y_tst, dtype=int).flatten()                               # Organiza os rotulos da amostra de testes
        y_ = num.array(classificador.predicao(X_tst), dtype=int).flatten()      # Calcula a predição da amostra de testes
        rodada += 1
        if len(num.unique(y_tst)) == 2:                                         # Testa se é uma classificação (apenas dois rótulos)
            matriz = classificador.gerarMatrizConfusao(y, y_)                   # Gera a matriz de confusão
            # print(matriz)
            VN = int(matriz.loc[1].loc[1])                                      # Valores encontrados como VERDADEIROS NEGATIVOS
            VP = int(matriz.loc[-1].loc[-1])                                    # Valores encontrados como VERDADEIROS POSITIVOS
            FN = int(matriz.loc[-1].loc[1])                                     # Valores encontrados como FALSOS NEGATIVOS
            FP = int(matriz.loc[1].loc[-1])                                     # Valores encontrados como FALSOS POSITIVOS
            # print("VN =", VN)
            # print("VP =", VP)
            # print("FN =", FN)
            # print("FP =", FP)                                                 # Calcula as medidas de desempenho:
            acuracia = (VP + VN) / (VP + VN + FP + FN)                          # Acurácia
            sensibilidade = VP / (VP + FN)                                      # Sensibilidade
            especificidade = VN / (VN + FP)                                     # Especificidade
            dados_rodada.append({
                "rodada": rodada,
                "desempenho": acuracia,
                "acuracia": acuracia,
                "sensibilidade": sensibilidade,
                "especificidade": especificidade,
                "pesos": classificador.getPesos().T,
                "matriz_confusao": matriz
            })                                                                  # Armazena os dados da rodada
        else:
            eqm = classificador.calcularEQM(y, y_)                              # Calcula o desempenho: EQM
            dados_rodada.append({
                "rodada": rodada,
                "desempenho": eqm,
                "eqm": eqm,
                "pesos": classificador.getPesos().T
            })                                                                  # Armazena os dados da rodada
    printProgressBar(100, 'Concluído !!!')
    dados = pd.DataFrame(dados_rodada)                                          # Organiza os dados para manipulação
    if "acuracia" in dados.columns:                                             # Se encontrar a coluna 'acuracia' no DataFrame, pega os dados referente a classificação
        calculos.append({
            nomeClassificador: {
                "acuracia": [
                    "{:.4f}".format(num.mean(dados.iloc[:]["acuracia"])),
                    "{:.4f}".format(num.median(dados.iloc[:]["acuracia"])),
                    "{:.4f}".format(num.min(dados.iloc[:]["acuracia"])),
                    "{:.4f}".format(num.max(dados.iloc[:]["acuracia"])),
                    "{:.4f}".format(num.std(dados.iloc[:]["acuracia"]))
                ],
                "especificidade": [
                    "{:.4f}".format(num.mean(dados.iloc[:]["especificidade"])),
                    "{:.4f}".format(num.median(dados.iloc[:]["especificidade"])),
                    "{:.4f}".format(num.min(dados.iloc[:]["especificidade"])),
                    "{:.4f}".format(num.max(dados.iloc[:]["especificidade"])),
                    "{:.4f}".format(num.std(dados.iloc[:]["especificidade"]))
                ],
                "sensibilidade": [
                    "{:.4f}".format(num.mean(dados.iloc[:]["sensibilidade"])),
                    "{:.4f}".format(num.median(dados.iloc[:]["sensibilidade"])),
                    "{:.4f}".format(num.min(dados.iloc[:]["sensibilidade"])),
                    "{:.4f}".format(num.max(dados.iloc[:]["sensibilidade"])),
                    "{:.4f}".format(num.std(dados.iloc[:]["sensibilidade"]))
                ]
            }
        })
    else:                                                                       # Se não encontrar, pega os dados referente a uma regressão
        calculos.append({
            nomeClassificador: {
                "eqm": [
                    "{:.4f}".format(num.mean(dados.iloc[:]["eqm"])),
                    "{:.4f}".format(num.median(dados.iloc[:]["eqm"])),
                    "{:.4f}".format(num.min(dados.iloc[:]["eqm"])),
                    "{:.4f}".format(num.max(dados.iloc[:]["eqm"])),
                    "{:.4f}".format(num.std(dados.iloc[:]["eqm"]))
                ]
            }
        })
    resultados = pd.DataFrame(calculos)                                         # Organiza os resultados para apresentação como saída do programa
    print("\n")
    msg = (f"Fonte de dados:\n {inputData.Name}")
    msg += ("\n")
    msg += (f"\nResultados:\n {resultados.keys().values[0]}")
    msg += (f"\n{str(pd.DataFrame(data=resultados.iloc[:][nomeClassificador][0], index=['media', 'mediana', 'minimo', 'maximo', 'd.padrao']).T)}")
    print(msg)
    # print("dados_rodada\n", dados_rodada)
    # print("dados\n", dados)
    # print("calculos\n", calculos)
    # print("resultados\n", resultados)
    # Mudar a pasta para guardar os arquivos gerados como resultado do programa
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), "resultados"))
    # Plota o gráfico de convergência do classificador
    plot.title('Convergência')
    plot.plot(classificador.getCustos()[::-1])
    plot.ylabel('Erros')
    plot.xlabel('Épocas')
    plot.savefig('%s_%s-Convergencia.png' % (nomeClassificador, inputData.Name))# Cria um arquivo '.png' com o gráfico de convergência
    plot.show()
    # Plota o gráfico de desempenho do classificador
    plot.title('Desempenho')
    plot.plot(dados['desempenho'][::-1])
    plot.ylabel('Valores')
    plot.xlabel('Épocas')
    plot.savefig('%s_%s-Desempenho.png' % (nomeClassificador, inputData.Name))  # Cria um arquivo '.png' con o gráfico de desempenho
    plot.show()
    # Cria um arquivo '.txt' com os resultados do programa
    with open('%s_%s-Resultados.txt' % (nomeClassificador, inputData.Name), 'w') as arquivo:
        arquivo.write(msg)
    # Cria um arquivo '.csv' com a tabela que armazena os pesos de cada rodada
    dados[['rodada', 'pesos']].to_csv('%s_%s-Pesos.csv' % (nomeClassificador, inputData.Name), index=False)

def close() -> None:
    sys.exit(0)

try:
    if __name__ == "__main__":
        # Carregando os dados
        espiral = loadData(file_name="spiral.csv", columns=["x1", "x2", "y",], separator=",", ignore_header=False)
        espiral.Name = "espiral"
        aerogerador = loadData("aerogerador.dat", ["Vel", "Pot"], "\t", False)
        aerogerador.Name = "aerogerador"
        red_wine = loadData("winequality-red.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        red_wine.Name = "red_wine"
        white_wine = loadData("winequality-white.csv", ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"], ";", True)
        white_wine.Name = "white_wine"
        # Inicialização dos classificadores com as taxas de aprendizado e o número de épocas para iterações de cada um
        percecptron = Perceptron(tx_aprendizado=0.01, n_iteracoes=100)
        adaline = Adaline(tx_aprendizado=0.0001, n_iteracoes=10)
        mlp = MultilayerPerceptron(tx_aprendizado=0.0001, n_iteracoes=10, n_camadas=3)
        clearScreen()                                                           # Limpa a tela
        # Perceptron                                                            # Classificador concluído
        # run(inputData=espiral, algoritmo=percecptron)
        run(inputData=aerogerador, algoritmo=percecptron)
        # run(inputData=red_wine, algoritmo=percecptron)
        # run(inputData=white_wine, algoritmo=percecptron)
        # Adaline                                                               # Classificador concluído
        # run(inputData=espiral, algoritmo=adaline)
        # run(inputData=aerogerador, algoritmo=adaline)
        # run(inputData=red_wine, algoritmo=adaline)
        # run(inputData=white_wine, algoritmo=adaline)
        # MultilayerPerceptron                                                  # Classificador em desenvolvimento, não implementado por completo
        # run(inputData=espiral, algoritmo=mlp)
        # run(inputData=aerogerador, algoritmo=mlp)
        # run(inputData=red_wine, algoritmo=mlp)
        # run(inputData=white_wine, algoritmo=mlp)

finally:
    close()
