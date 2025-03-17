import numpy as np
from DenseLayer import DenseLayer, ReLU, Softmax, CrossEntropy
from MnistDataset import MnistDataset

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.capa1 = DenseLayer(input_size, hidden_size)
        self.activation1 = ReLU()
        self.capa2 = DenseLayer(hidden_size, output_size)
        self.activation2 = Softmax()
        self.loss_function = CrossEntropy()
        self.learning_rate = learning_rate

        self.capa1.cargar_pesos("Mnist/pesosguardados")
        self.capa2.cargar_pesos("Mnist/pesosguardados")

    def forward(self, X):
        self.z1 = self.capa1.forward(X)
        self.a1 = self.activation1.forward(self.z1)
        self.z2 = self.capa2.forward(self.a1)
        self.a2 = self.activation2.forward(self.z2)
        return self.a2

    def backward(self, X, y_true, y_pred):
        grad_loss = self.loss_function.backward(y_pred, y_true)
        grad_a2 = self.activation2.backward(grad_loss)
        grad_z2 = self.capa2.backward(grad_a2)
        grad_a1 = self.activation1.backward(grad_z2)
        self.capa1.backward(grad_a1)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss_function.forward(y_pred, y)
            self.backward(X, y, y_pred)

            print(f"Epoch [{epoch}] ---- Loss: [{loss:.4f}]")
            self.capa1.guardar_pesos("Mnist/pesosguardados")
            self.capa2.guardar_pesos("Mnist/pesosguardados")

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)
