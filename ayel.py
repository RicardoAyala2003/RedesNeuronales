import numpy as np

class capaDensa:
    def _init_(self, entradas: int, neuronas: int):
        self.pesos = np.random.randn(entradas,neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))

    def forward(self, datos:list[float]):
        self.salida = np.matmul(datos,self.pesos) + self.sesgos

class ReLU:
    def forward(self,x:list[float]):
            self.salida= np.maximum(0,x)

class Softmax:
    def forward(self,x:list[float]):
            exp_x = np.exp(x-np.max(x))
            self.salida = exp_x / np.sum(exp_x,axis=1)


  

capa1 = capaDensa(5,10)

capa2 = capaDensa(10,10)

capaSalida = capaDensa(10,4)

relu1 = ReLU()
relu2 = ReLU()
softmax_salida= Softmax()

entradasEj=[1,2,3,4,5]
capa1.forward(entradasEj)
relu1.forward(capa1.salida)

capa2.forward(relu1.salida)
relu2.forward(capa2.salida)

capaSalida.forward(relu2.salida)
softmax_salida.forward(capaSalida.salida)
print(capaSalida.salida)
print(softmax_salida.salida)