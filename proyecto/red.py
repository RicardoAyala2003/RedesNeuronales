import numpy as np
from DenseLayer import DenseLayer, ReLU, Softmax, CrossEntropy

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.0005, optimizer=None):
        self.capa1 = DenseLayer(input_size, hidden_size1)
        self.capa1.weights_loader("Mnist/pesosguardadosc1")
        self.activation1 = ReLU()
        
        self.capa2 = DenseLayer(hidden_size1, hidden_size2)
        self.capa2.weights_loader("Mnist/pesosguardadosc2")
        self.activation2 = ReLU()
        
        self.capa3 = DenseLayer(hidden_size2, output_size)
        self.capa3.weights_loader("Mnist/pesosguardadosc3")
        self.activation3 = Softmax()
        
        self.loss_function = CrossEntropy()
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        
        self.training_loss = []
        self.test_accuracy = []
    
    def forward(self, X):
        self.z1 = self.capa1.forward(X)
        self.a1 = self.activation1.forward(self.z1)
        
        self.z2 = self.capa2.forward(self.a1)
        self.a2 = self.activation2.forward(self.z2)
        
        self.z3 = self.capa3.forward(self.a2)
        self.a3 = self.activation3.forward(self.z3)
        return self.a3
    
    def backward(self, X, y_true, y_pred, lambda_l2=0.0001):
        grad_loss = self.loss_function.compute_gradient(y_true, y_pred)
        grad_a3 = self.activation3.backward(grad_loss, y_pred)
        grad_z3 = self.capa3.backward(grad_a3, lambda_l2)  
        
        grad_a2 = self.activation2.backward(grad_z3)
        grad_z2 = self.capa2.backward(grad_a2, lambda_l2)
        
        grad_a1 = self.activation1.backward(grad_z2)
        self.capa1.backward(grad_a1, lambda_l2) 
    
    def train(self, X, y, epochs, batch_size, ytest, X_test, saveandprinteach):
        num_samples = X.shape[0]
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            epoch_loss = 0
            num_batches = 0 
            
            for i in range(0, num_samples, batch_size):
                batch_X = X[indices[i:i+batch_size]]
                batch_y = y[indices[i:i+batch_size]]
                
                y_pred = self.forward(batch_X)
                loss = self.loss_function.compute_loss(batch_y, y_pred)
                self.backward(batch_X, batch_y, y_pred)
                
                epoch_loss += loss
                num_batches += 1 
                
                if self.optimizer is not None:
                    self.optimizer.pre_update_params()
                    self.optimizer.update_params(self.capa1)
                    self.optimizer.update_params(self.capa2)
                    self.optimizer.update_params(self.capa3)
                    self.optimizer.post_update_params()
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            self.training_loss.append(avg_epoch_loss)
            
            y_test_pred = self.predict(X_test)
            accuracy = np.mean(np.argmax(ytest, axis=1) == y_test_pred)
            self.test_accuracy.append(accuracy)
            
            if epoch % saveandprinteach == 0:
                print("=" * 40)
                print(f" Epoch: {epoch:03d}")
                print(f" Average Loss: {avg_epoch_loss:.4f}")
                print(f" Test Accuracy: {accuracy*100:.2f}%")
                print("=" * 40)
                self.capa1.weights_saver("Mnist/pesosguardadosc1")
                self.capa2.weights_saver("Mnist/pesosguardadosc2")
                self.capa3.weights_saver("Mnist/pesosguardadosc3")
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)