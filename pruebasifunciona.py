import numpy as np

class capaDensa:
    def __init__(self, entradas: int, neuronas: int):
        self.pesos = np.random.randn(entradas,neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))

    def forward(self, datos):
        self.salida = np.dot(datos,self.pesos) + self.sesgos
        
