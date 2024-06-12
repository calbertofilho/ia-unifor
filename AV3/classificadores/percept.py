import numpy as np

# Um perceptron para 4 inputs
# os inputs são as colunas do dataset de treino
class Perceptron(object):
    """Essa classe consiste em um perceptron: um modelo
       linear concedido por  Frank Rosenblatt.
       Essa classe é apenas para classificação.
       alpha: defaut=0.01 # A taxa em que o erro será propagado.
       n_features: O número de features no seu dataset.
       n_iter: O número de iterações realizadas pelo perceptron."""

    def __init__(self, alpha=0.01, n_features=3, n_iter=2000):
        self.w = np.random.uniform(-1, 1, n_features+1) # inicializa os pesos
        self.alpha = alpha                              # a taxa de aprendizado
        self.n_iter = n_iter                            # o número de iterações

    def sign(self, num): # Função que avalia o output
        return 1. if (num >= 0.0) else -1.

    def training(self, X, y):
        """Esse método é usado para treinar o perceptron.
           x_data: Uma numpy.array contendo as features para treino.
           y_data: Uma numpy.array contendo as classes(target)"""
        X = np.concatenate((X, np.ones((len(X.T[0]), 1))), axis=1)  # acrescentamos o bias ao dataset, no caso, mais uma coluna contendo apenas 1
        for i in range(self.n_iter):
            cum_erro = 0                            # Aqui armazenamos o erro acumulado para parar a otimização
            for j in range(len(X)):
                output = self.w.dot(X[j])           # O output é o produto dos pesos pela linha atual
                if self.sign(output) != y[j]:       # avaliamos para ver se é correspondente.
                    cum_erro += 1                   # Caso não seja acrescentamos a contagem de erro
                    erro = y[j] - output            # medimos o erro da iteração de forma direta.(sem loss)
                    self.w += self.alpha*erro*X[j]  # Aqui os pesos são atualizados
            if cum_erro == 0:                       # Aqui avaliamos o erro acumulado caso sejá 0 para o treinamento.
                break                

    def predict(self, vector):
        """O método predict pode levar uma numpy.array de uma ou duas
           dimensões."""
        if np.ndim(vector) == 1:                        # Avaliamos a quantidade de dimensões.
            vector = np.insert(vector, len(vector), 1)  # inserimos o bias
            prediction = self.sign(self.w.dot(vector))  # Fazemos a predição.
        else:                                           # Caso contrário é feito o mesmo processo porém com uma array de duas dimensões.
            vector = np.insert(vector[:,], len(vector[0]), 1, axis=1)
            prediction = [[self.sign(self.w.dot(x))] for x in vector]
        return np.array(prediction)

    def showAccuracy(self, y_true, predictions):
        correct = 0
        for x in range(len(y_true)):
            if y_true[x] == predictions[x]:     # Compara os respectivos valores
                correct += 1
        return float(correct) / len(y_true)
