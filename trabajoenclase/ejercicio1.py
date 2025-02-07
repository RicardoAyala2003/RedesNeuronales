import numpy as np
import sys

class CapaDensa:
    def __init__(self, entradas: int, neuronas: int):
        self.pesos = np.random.randn(entradas, neuronas) * 0.01
        self.sesgos = np.zeros((1, neuronas))

    def adelante(self, datos):
        self.salida = np.dot(datos, self.pesos) + self.sesgos

class Sigmoide:
    def adelante(self, x):
        self.salida = 1 / (1 + np.exp(-x))

class ReLU:
    def adelante(self, x):
        self.salida = np.maximum(0, x)

def caracter_a_numero(caracter):
    return ord(caracter.lower()) - ord('a') + 1 if caracter.isalpha() else 0

def convertir_a_vector_numerico(palabra):
    palabra = palabra.ljust(8, '0')
    return [caracter_a_numero(caracter) for caracter in palabra]

def detector_de_palindromos(palabra):
    vector = convertir_a_vector_numerico(palabra)
    capa_oculta1 = CapaDensa(8, 8)
    capa_oculta1.adelante(np.array([vector]))
    sigmoide1 = Sigmoide()
    sigmoide1.adelante(capa_oculta1.salida)
    
    capa_oculta2 = CapaDensa(8, 8)
    capa_oculta2.adelante(sigmoide1.salida)
    sigmoide2 = Sigmoide()
    sigmoide2.adelante(capa_oculta2.salida)
    
    capa_salida = CapaDensa(8, 1)
    capa_salida.adelante(sigmoide2.salida)
    relu_salida = ReLU()
    relu_salida.adelante(capa_salida.salida)
    
    salida_ajustada = relu_salida.salida
    return salida_ajustada

if len(sys.argv) != 2:
    palabra = input("Ingrese una palabra: ")
else:
    palabra = sys.argv[1]

vector_numerico = convertir_a_vector_numerico(palabra)
salida_red = detector_de_palindromos(palabra)

print(f"{palabra}")
print(f"{vector_numerico}")
print(salida_red)
