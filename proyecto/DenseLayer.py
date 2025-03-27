import numpy as np
import os

class DenseLayer:
    def __init__(self, input_size, output_size):
        # Inicializa pesos con distribución normal escalada y sesgos en ceros
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        # Almacena las entradas y calcula la salida (producto punto + sesgos)
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, grad_output, lambda_l2=0.0): 
        # Calcula gradientes de pesos (con regularización L2 opcional), sesgos y entrada
        self.dweights = np.dot(self.inputs.T, grad_output) + 2 * lambda_l2 * self.weights
        self.dbiases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

    def weights_loader(self, path):
        # Intenta cargar pesos y sesgos desde archivos .npy en la ruta dada
        try:
            loaded_weights = np.load(f"{path}/weights.npy")
            loaded_biases = np.load(f"{path}/biases.npy")

            # Verifica dimensiones antes de asignar
            if loaded_weights.shape == self.weights.shape and loaded_biases.shape == self.biases.shape:
                self.weights = loaded_weights
                self.biases = loaded_biases
            else:
                print(f"Error de dimensión en {path}.")

        except FileNotFoundError:
            # Maneja caso de archivos no encontrados
            print(f"No se encontraron pesos en {path}.")

    def weights_saver(self, path):
        # Crea directorio si no existe y guarda pesos y sesgos en archivos .npy
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f"{path}/weights.npy", self.weights)
        np.save(f"{path}/biases.npy", self.biases)
        print("Pesos y sesgos guardados")


class ReLU:
    def forward(self, inputs):
        # Aplica ReLU (valores negativos a 0) y guarda entradas
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        # Calcula gradiente de ReLU (0 si entrada <= 0)
        grad = grad_output.copy()
        grad[self.inputs <= 0] = 0
        return grad

class Softmax:
    def forward(self, inputs):
        # Calcula Softmax estabilizado numéricamente para probabilidades
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)
        return exp_values / (sum_exp_values + 1e-9)

    def backward(self, grad_output, outputs):
        # Devuelve gradiente directamente (típico en Softmax con entropía cruzada)
        return grad_output


class CrossEntropy:
    def compute_loss(self, y_true, y_pred):
        # Calcula pérdida de entropía cruzada promedio
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def compute_gradient(self, y_true, y_pred):
        # Calcula gradiente de la pérdida respecto a las predicciones
        return y_pred - y_true