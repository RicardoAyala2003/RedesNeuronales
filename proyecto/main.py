from MnistDataset import MnistDataset
from red import NeuralNetwork
from OptimizerAdam import Optimizer_Adam
import numpy as np
import matplotlib.pyplot as plt

# Configuración 
INPUT_SIZE = 784
HIDDEN_SIZE1 = 128   
HIDDEN_SIZE2 = 64    
OUTPUT_SIZE = 10
LEARNING_RATE = 0.0005
EPOCHS = 20
BATCH_SIZE = 64
SAVE_AND_PRINT_EACH = 1
NUM_SAMPLES = 3
SHOW_RESULTS = True

def load_data():
    # Carga y prepara los datos MNIST.
    train_dataset = MnistDataset()
    train_dataset.load("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte")
    test_dataset = MnistDataset()
    test_dataset.load("dataset/t10k-images-idx3-ubyte", "dataset/t10k-labels-idx1-ubyte")

    train_images = train_dataset.get_flattened_data()
    train_labels = train_dataset.get_one_hot_labels()
    test_images = test_dataset.get_flattened_data()
    test_labels = test_dataset.get_one_hot_labels()

    return train_images, train_labels, test_images, test_labels

def display_sample_predictions(model, images, labels, num_samples=NUM_SAMPLES):
    # Muestra predicciones de muestras aleatorias.
    random_indices = np.random.choice(len(images), num_samples)
    
    for index in random_indices:
        sample_image = images[index].reshape(28, 28)
        predicted_label = model.predict(images[index].reshape(1, -1))[0]
        actual_label = np.argmax(labels[index])
        
        plt.figure(figsize=(4, 4))
        plt.imshow(sample_image, cmap='gray')
        plt.title(f"Real: {actual_label} | Pred: {predicted_label}")
        plt.axis('off')
        if SHOW_RESULTS:
            plt.show()
        else:
            plt.savefig(f"sample_prediction_{index}.png")  
            plt.close()

def plot_metrics(network):
    # Grafica la pérdida y la precisión durante el entrenamiento.
    plt.figure(figsize=(12, 5))
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(network.training_loss, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(network.test_accuracy, label='Test Accuracy', color='orange')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if SHOW_RESULTS:
        plt.show()
    else:
        plt.savefig("training_metrics.png")  
        plt.close()

def train_and_evaluate_model():
    # Entrena y evalúa la red neuronal y cargar datos
    train_images, train_labels, test_images, test_labels = load_data()

    adam_optimizer = Optimizer_Adam(learning_rate=LEARNING_RATE, decay=1e-3)
    
    # Cambiar la inicialización para que se ajuste a la red de tres capas 
    neural_net = NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE1, HIDDEN_SIZE2, OUTPUT_SIZE, LEARNING_RATE, adam_optimizer)

  
    neural_net.train(train_images, train_labels, EPOCHS, BATCH_SIZE, test_labels, test_images, SAVE_AND_PRINT_EACH)

   
    predicted_labels = neural_net.predict(test_images)
    accuracy = np.mean(np.argmax(test_labels, axis=1) == predicted_labels)
    print(f"Accuracy: [{accuracy * 100:.2f}%]")

    # Mostrar resultados
    if SHOW_RESULTS:
        display_sample_predictions(neural_net, test_images, test_labels)
        plot_metrics(neural_net)
    else:
        # Guardar gráficas si no se muestran
        display_sample_predictions(neural_net, test_images, test_labels)
        plot_metrics(neural_net)

def main():
   
    try:
        train_and_evaluate_model()
    except Exception as e:
        print(f"Error durante el entrenamiento o evaluación: {e}")

if __name__ == "__main__":
    main()
