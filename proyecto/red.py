import numpy as np
from DenseLayer import DenseLayer, ReLU, Softmax, CrossEntropy
from train import train  # Importamos la función train

class NeuralNetwork:
    def __init__(self, input_size, size1, size2, output_size, learning_rate=0.0005, optimizer=None):
        # Crea primera capa y carga pesos desde archivo
        self.capa1 = DenseLayer(input_size, size1)
        self.capa1.weights_loader("Mnist/pesosguardadosc1")
        self.activation1 = ReLU()
        
        # Crea segunda capa y carga pesos desde archivo
        self.capa2 = DenseLayer(size1, size2)
        self.capa2.weights_loader("Mnist/pesosguardadosc2")
        self.activation2 = ReLU()
        
        # Crea tercera capa y carga pesos desde archivo
        self.capa3 = DenseLayer(size2, output_size)
        self.capa3.weights_loader("Mnist/pesosguardadosc3")
        self.activation3 = Softmax()
        
        # Establece función de pérdida y parámetros
        self.loss_function = CrossEntropy()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        
        # Inicializa listas para métricas
        self.training_loss = []
        self.test_accuracy = []
    
    def forward(self, X):
        # Propagación forward capa 1
        self.z1 = self.capa1.forward(X)
        self.a1 = self.activation1.forward(self.z1)
        
        # Propagación forward capa 2
        self.z2 = self.capa2.forward(self.a1)
        self.a2 = self.activation2.forward(self.z2)
        
        # Propagación forward capa 3
        self.z3 = self.capa3.forward(self.a2)
        self.a3 = self.activation3.forward(self.z3)
        return self.a3
    
    def backward(self, X, y_true, y_pred, lambda_l2=0.0001):
        # Gradiente de la pérdida y backward capa 3
        grad_loss = self.loss_function.compute_gradient(y_true, y_pred)
        grad_a3 = self.activation3.backward(grad_loss, y_pred)
        grad_z3 = self.capa3.backward(grad_a3, lambda_l2)  
        
        # Backward capa 2
        grad_a2 = self.activation2.backward(grad_z3)
        grad_z2 = self.capa2.backward(grad_a2, lambda_l2)
        
        # Backward capa 1
        grad_a1 = self.activation1.backward(grad_z2)
        self.capa1.backward(grad_a1, lambda_l2) 
    
    def train(self, X, y, epochs, batch_size, ytest, X_test, saveandprinteach):
        # Ejecuta función de entrenamiento importada
        train(self, X, y, epochs, batch_size, ytest, X_test, saveandprinteach)  # Llamamos la función importada
    
    def predict(self, X):
        # Predice clase con mayor valor
        return np.argmax(self.forward(X), axis=1)