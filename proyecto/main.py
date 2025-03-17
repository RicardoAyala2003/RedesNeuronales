from MnistDataset import MnistDataset
from red import NeuralNetwork  
import numpy as np

def main():
    print("Cargando datos de entrenamiento...")
    
   
    mnist_train = MnistDataset()
    mnist_train.load("/home/ricardo/Redes/Mnist/dataset/train-images-idx3-ubyte", 
                     "/home/ricardo/Redes/Mnist/dataset/train-labels-idx1-ubyte")
    
    mnist_test = MnistDataset()
    mnist_test.load("/home/ricardo/Redes/Mnist/dataset/t10k-images-idx3-ubyte", 
                    "/home/ricardo/Redes/Mnist/dataset/t10k-labels-idx1-ubyte")

    X_train = mnist_train.get_flattened_data()
    y_train = mnist_train.get_one_hot_labels()
    X_test = mnist_test.get_flattened_data()
    y_test = mnist_test.get_one_hot_labels()

    
    input_size = 784  
    hidden_size = 128  
    output_size = 10  
    learning_rate = 0.1  

    
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    nn.train(X_train, y_train, epochs=10)

    y_test_pred = nn.predict(X_test)
    accuracy = np.mean(np.argmax(y_test, axis=1) == y_test_pred)
    
    print(f"Accuracy: [{accuracy * 100:.2f}%]")

if __name__ == "__main__":
    main()
