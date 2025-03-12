import numpy as np
  
class capaDensa:
    def __init__(self, entradas: int, neuronas: int):
        self.pesos = np.random.randn(entradas, neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))

    def forward(self, datos: list[float]):
        self.entrada = datos
        self.salida = np.matmul(datos, self.pesos) + self.sesgos

    def backward(self, dvalues: list[float]):
        self.dpesos = np.dot(self.entrada.T, dvalues)
        self.dsesgos = np.sum(dvalues, axis=0, keepdims=True)
        self.dentrada = np.dot(dvalues, self.pesos.T)
        return self.dentrada

    def update(self, learning_rate: float):
        self.pesos -= learning_rate * self.dpesos
        self.sesgos -= learning_rate * self.dsesgos

class ReLU:
    def forward(self, x: list[float]):
        self.entrada = x
        self.salida = np.maximum(0, x)

    def backward(self, dvalues: list[float]):
        self.dentrada = dvalues.copy()
        self.dentrada[self.entrada <= 0] = 0
        return self.dentrada

class Softmax:
    def forward(self, x: list[float]):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.salida = exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, dvalues: list[float]):
        self.dentrada = dvalues.copy()
        return self.dentrada

class Sigmoide:
    def forward(self, x: list[float]):
        self.entrada = x
        self.salida = 1 / (1 + np.exp(-x))

    def backward(self, dvalues: list[float]):
        self.dentrada = dvalues * (1 - self.salida) * self.salida
        return self.dentrada

class CrossEntropy:
    def forward(self, y_pred: list[float], y_true: list[float]):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        correct_confidences = y_pred_clipped[range(samples), y_true]
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)

    def backward(self, dvalues: list[float], y_true: list[float]):
        samples = len(dvalues)
        labels = len(dvalues[0])
        y_true = np.eye(labels)[y_true]
        self.dentrada = -y_true / dvalues
        self.dentrada = self.dentrada / samples
        return self.dentrada