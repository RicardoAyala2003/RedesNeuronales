import numpy as np
import os

class capaDensa:
    def __init__(self, entradas: int, neuronas: int):
        self.pesos = np.random.randn(entradas, neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))

    def forward(self, datos: np.ndarray):
        self.entrada = datos
        self.salida = np.dot(datos, self.pesos) + self.sesgos
        return self.salida  # Necesitamos retornar la salida aquí

    def backward(self, dvalues: np.ndarray):
        self.dpesos = np.dot(self.entrada.T, dvalues)
        self.dsesgos = np.sum(dvalues, axis=0, keepdims=True)
        self.dentrada = np.dot(dvalues, self.pesos.T)
        return self.dentrada

    def update(self, learning_rate: float):
        self.pesos -= learning_rate * self.dpesos
        self.sesgos -= learning_rate * self.dsesgos

    def guardar_pesos(self, path="Mnist/pesosguardados"):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f"{path}/pesos.npy", self.pesos)
        np.save(f"{path}/sesgos.npy", self.sesgos)
        print("Pesos y sesgos guardados.")

    def cargar_pesos(self, path="Mnist/pesosguardados"):
        try:
            self.pesos = np.load(f"{path}/pesos.npy")
            self.sesgos = np.load(f"{path}/sesgos.npy")
            print("Pesos y sesgos cargados.")
        except FileNotFoundError:
            print(" error: No se encontraron los archivos de pesos y sesgos.")


class ReLU:
    def forward(self, x: np.ndarray):
        self.entrada = x
        self.salida = np.maximum(0, x)

    def backward(self, dvalues: np.ndarray):
        self.dentrada = dvalues.copy()
        self.dentrada[self.entrada <= 0] = 0
        return self.dentrada

class Softmax:
    def forward(self, inputs: np.ndarray):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)
        self.output = exp_values / sum_exp_values
        return self.output

    def backward(self, grad_output: np.ndarray):
        return grad_output

class CrossEntropy:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.sum(y_true * np.log(y_pred_clipped), axis=1)
        return np.mean(loss)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        samples = len(y_pred)
        return (y_pred - y_true) / samples
