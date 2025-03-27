import numpy as np

def train(network, X, y, epochs, batch_size, ytest, X_test, saveandprinteach):
    # Obtiene número de muestras
    num_samples = X.shape[0]
    
    # Muestra pesos iniciales de las capas (primeras 5x5)
    print("Pesos antes del entrenamiento:")
    print("Capa 1:", network.capa1.weights[:5, :5])
    print("Capa 2:", network.capa2.weights[:5, :5])
    print("Capa 3:", network.capa3.weights[:5, :5])
    
    # Ciclo por épocas
    for epoch in range(epochs):
        # Permuta índices para batches aleatorios
        indices = np.random.permutation(num_samples)
        epoch_loss = 0
        num_batches = 0 
        
        # Procesa batches
        for i in range(0, num_samples, batch_size):
            # Extrae batch de datos y etiquetas
            batch_X = X[indices[i:i+batch_size]]
            batch_y = y[indices[i:i+batch_size]]
            
            # Calcula predicciones y pérdida
            y_pred = network.forward(batch_X)
            loss = network.loss_function.compute_loss(batch_y, y_pred)
            # Propagación hacia atrás
            network.backward(batch_X, batch_y, y_pred)
            
            # Acumula pérdida y cuenta batches
            epoch_loss += loss
            num_batches += 1 
            
            # Actualiza parámetros si hay optimizador
            if network.optimizer is not None:
                network.optimizer.pre_update_params()
                network.optimizer.update_params(network.capa1)
                network.optimizer.update_params(network.capa2)
                network.optimizer.update_params(network.capa3)
                network.optimizer.post_update_params()
        
        # Calcula pérdida promedio por época
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        network.training_loss.append(avg_epoch_loss)
        
        # Evalúa precisión en datos de prueba
        y_test_pred = network.predict(X_test)
        accuracy = np.mean(np.argmax(ytest, axis=1) == y_test_pred)
        network.test_accuracy.append(accuracy)
        
        # Imprime métricas y guarda pesos cada cierto intervalo
        if epoch % saveandprinteach == 0:
            print("=" * 40)
            print(f" Epoch: {epoch:03d}")
            print(f" Average Loss: {avg_epoch_loss:.4f}")
            print(f" Test Accuracy: {accuracy*100:.2f}%")
            print("=" * 40)
            network.capa1.weights_saver("Mnist/pesosguardadosc1")
            network.capa2.weights_saver("Mnist/pesosguardadosc2")
            network.capa3.weights_saver("Mnist/pesosguardadosc3")
    
    # Muestra pesos finales de las capas (primeras 5x5)
    print("Pesos después del entrenamiento:")
    print("Capa 1:", network.capa1.weights[:5, :5])
    print("Capa 2:", network.capa2.weights[:5, :5])
    print("Capa 3:", network.capa3.weights[:5, :5])