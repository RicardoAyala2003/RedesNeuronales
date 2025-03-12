import numpy as np
from keras.datasets import mnist
from DenseLayer import capaDensa, ReLU, Softmax, CrossEntropy


def preprocess_data(x, y):
    x = x.reshape(x.shape[0], -1).astype('float32') / 255.0
    y = np.eye(10)[y]
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)
input_size = 784
hidden_neurons = 128
output_neurons = 10

dense1 = capaDensa(input_size, hidden_neurons)
activation1 = ReLU()
dense2 = capaDensa(hidden_neurons, output_neurons)
activation2 = Softmax()
loss_function = CrossEntropy()


learning_rate = 0.1
epochs = 10
batch_size = 64


for epoch in range(epochs):
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

     
        dense1.forward(x_batch)
        activation1.forward(dense1.salida)
        dense2.forward(activation1.salida)
        activation2.forward(dense2.salida)

        loss = loss_function.forward(activation2.salida, np.argmax(y_batch, axis=1))
        predictions = np.argmax(activation2.salida, axis=1)
        accuracy = np.mean(predictions == np.argmax(y_batch, axis=1))

        loss_function.backward(activation2.salida, np.argmax(y_batch, axis=1))
        dense2.backward(loss_function.dentrada)
        activation1.backward(dense2.dentrada)
        dense1.backward(activation1.dentrada)

        dense1.update(learning_rate)
        dense2.update(learning_rate)

    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')