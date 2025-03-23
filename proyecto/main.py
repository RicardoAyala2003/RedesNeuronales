from MnistDataset import MnistDataset
from red import NeuralNetwork
from OptimizerAdam import Optimizer_Adam
import numpy as np
import matplotlib.pyplot as plt

# Configuración 
EPOCHS = 10
BATCH_SIZE = 64
SAVE_AND_PRINT_EACH = 1
NUM_SAMPLES = 5
SHOW_RESULTS = True
INPUT_SIZE = 784
SIZE1 = 128   
SIZE2 = 64    
OUTPUT_SIZE = 10
LEARNING_RATE = 0.1


def load_data():
    # Carga y prepara los datos MNIST
    train_dataset = MnistDataset()
    train_dataset.load("dataset/train-images-idx3-ubyte", "dataset/train-labels-idx1-ubyte")
    test_dataset = MnistDataset()
    test_dataset.load("dataset/t10k-images-idx3-ubyte", "dataset/t10k-labels-idx1-ubyte")

    train_images = train_dataset.get_flattened_data()
    train_labels = train_dataset.get_one_hot_labels()
    test_images = test_dataset.get_flattened_data()
    test_labels = test_dataset.get_one_hot_labels()

    return train_images, train_labels, test_images, test_labels

def display_pred(model, images, labels, num_samples=NUM_SAMPLES):
    selected_indices = np.random.choice(len(images), num_samples)
    
    for idx in selected_indices:
        image_to_show = images[idx].reshape(28, 28)
        model_prediction = model.predict(images[idx].reshape(1, -1))[0]
        true_label = np.argmax(labels[idx])
        plt.figure(figsize=(4, 4))
        plt.imshow(image_to_show, cmap='gray')
        plt.title(f"Real: {true_label}\nPred: {model_prediction}", fontsize=12, pad=10)
        plt.axis('off')

        if SHOW_RESULTS:
            plt.show()
        else:
            plt.savefig(f"prediccion_muestra_{idx}.png")  
            plt.close()

def plot_metrics(network):
    # Grafica la pérdida y la precisión durante el entrenamiento.
    plt.figure(figsize=(14, 6))

    # Gráfica de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(network.training_loss, label='Pérdida en entrenamiento', color='blue', linewidth=2)
    plt.title('Evolución de la Pérdida', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Pérdida', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    # Gráfica de precisión
    plt.subplot(1, 2, 2)
    plt.plot(network.test_accuracy, label='Precisión en prueba', color='green', linewidth=2)
    plt.title('Precisión en Conjunto de Prueba', fontsize=14, fontweight='bold')
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Precisión (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    
    if SHOW_RESULTS:
        plt.show()
    else:
        plt.savefig("training_metrics.png", dpi=300)
        plt.close()

def redtraining():
    # Entrena y evalúa la red neuronal y cargar datos
    train_images, train_labels, test_images, test_labels = load_data()

    adam_optimizer = Optimizer_Adam(learning_rate=LEARNING_RATE, decay=1e-3)
    # Cambiar la inicialización para que se ajuste a la red de tres capas 
    neural_net = NeuralNetwork(INPUT_SIZE, SIZE1, SIZE2, OUTPUT_SIZE, LEARNING_RATE, adam_optimizer)
    neural_net.train(train_images, train_labels, EPOCHS, BATCH_SIZE, test_labels, test_images, SAVE_AND_PRINT_EACH)

    predicted_labels = neural_net.predict(test_images)
    accuracy = np.mean(np.argmax(test_labels, axis=1) == predicted_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Mostrar resultados
    if SHOW_RESULTS:
        display_pred(neural_net, test_images, test_labels)
        plot_metrics(neural_net)
    else:
        # Guardar gráficas si no se muestran
        display_pred(neural_net, test_images, test_labels)
        plot_metrics(neural_net)

def main():
   
    try:
        redtraining()
    except Exception as e:
        print(f"Error durante el entrenamiento o evaluación: {e}")

if __name__ == "__main__":
    main()
