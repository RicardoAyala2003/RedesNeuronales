import numpy as np
import os

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((1, output_size))

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.biases

    def backward(self, grad_output, lambda_l2=0.0): 
        self.dweights = np.dot(self.inputs.T, grad_output) + lambda_l2 * self.weights
        self.dbiases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases

    def weights_saver(self, path="Mnist/pesosguardados"):
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(f"{path}/weights.npy", self.weights)
        np.save(f"{path}/biases.npy", self.biases)
        print("Pesos y sesgos guardados correctamente.")

    def weights_loader(self, path="Mnist/pesosguardados"):
        try:
            loaded_weights = np.load(f"{path}/weights.npy")
            loaded_biases = np.load(f"{path}/biases.npy")

            if loaded_weights.shape == self.weights.shape and loaded_biases.shape == self.biases.shape:
                self.weights = loaded_weights
                self.biases = loaded_biases
                print(f"Pesos cargados desde {path}")
            else:
                print(f"Error de dimensión en {path}. Se usará inicialización aleatoria.")

        except FileNotFoundError:
            print(f"No se encontraron pesos en {path}. Se usará inicialización aleatoria.")

                        
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.inputs <= 0] = 0
        return grad

class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        sum_exp_values = np.sum(exp_values, axis=1, keepdims=True)
        return exp_values / (sum_exp_values + 1e-9)

    def backward(self, grad_output, outputs):
        return grad_output


class CrossEntropy:
    def compute_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def compute_gradient(self, y_true, y_pred):
        return y_pred - y_true