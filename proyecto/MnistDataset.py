import numpy as np
import struct

class MnistDataset:
    def __init__(self):
        self.images = None
        self.labels = None

    def load(self, images_filename, labels_filename):
        with open(images_filename, 'rb') as f:
            magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
            if magic != 2051:
                raise ValueError(f"Número mágico incorrecto en imágenes: {magic}")
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols) / 255.0
        
        with open(labels_filename, 'rb') as f:
            magic, num_labels = struct.unpack('>II', f.read(8))
            if magic != 2049:
                raise ValueError(f"Número mágico incorrecto en etiquetas: {magic}")
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

    def get_flattened_data(self):
        return self.images

    def get_one_hot_labels(self, num_classes=10):
        return np.eye(num_classes)[self.labels]