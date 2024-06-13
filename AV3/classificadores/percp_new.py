import numpy as np

class Perceptron(object):

    def __init__(self, tx_aprendizado = 0.001, n_iteracoes = 100):
        # Construtor da classe
        self.eta = tx_aprendizado
        self.epocas = n_iteracoes
        self.pesos = None
        #self.bias = None
        self.bias = -1

    def ativacao(self, valor_entrada):
        # Função de ativação
        return np.where(valor_entrada >= 0, 1, -1)

    def treinamento(self, X, y):
        # Funcao de treinamento
        erro = True
        epoca = 0
        qtde_amostras, qtde_caracteristicas = X.shape
        self.pesos = np.random.uniform(size = qtde_caracteristicas, low = -0.5, high = 0.5)
        #self.bias = np.random.uniform(low = -0.5, high = 0.5)
        while erro and (epoca < self.epocas):
            erro = False
            for indice, caracteristicas in enumerate(X):
                resultado = np.dot(caracteristicas, self.pesos) + self.bias
                y_predito = self.ativacao(resultado)
                self._atualiza_pesos(caracteristicas, y[indice], y_predito)
                # print("indice =", indice)
                # print("u_t =", resultado)
                # print("y_t =", y_predito)
                if y_predito != y[indice]:
                    erro = True
            epoca += 1

    def _atualiza_pesos(self, amostra, y_atl, y_pred):
        # Funcao que atualiza os pesos
        erro = y_atl - y_pred
        correcao = self.eta * erro
        self.pesos += correcao * amostra
        #self.bias += correcao
        # print("e_t =", erro)
        # print("correcao =", correcao)
        # print("pesos =", self.pesos)
        # print("bias =", self.bias)

    def predicao(self, amostras_teste):
        # Funcao de teste
        resultado = np.dot(amostras_teste, self.pesos) + self.bias
        y_predito = self.ativacao(resultado)
        return y_predito
