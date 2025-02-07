import numpy as np
name = "juan"
edad = 35

print("Hola, me llamo",name)
print("Tengo", edad, "años")

city= "San Pedro Sula"
current_year= "2025"

print("Vivo en", city,"en el anio", current_year)

print(10//2) #entero
print(10/2) #decimal

nombre= "juan michaeljackson turcios"
edad2=28
vacio=""
nombreEdad=nombre+" " +str(edad2)
print(nombreEdad)

repetido = nombre*5
print(repetido)

sliceNombre= nombre[-7:]
print(sliceNombre)


nombre = "Leonardo da Vinci"
edad = 67
saludo = f"Hola, mi nombre es {nombre} y tengo {edad} años."
print(saludo)


entero1= 120
entero2=40

print(entero1+entero2)
print(entero1-entero2)
print(entero1*entero2)
print(entero1//entero2)
print(entero1%entero2)

float1=1.5
float2=6.5

print(float1+float2)
print(float1-float2)
print(float1*float2)
print(float1/float2)
print(float1%float2)

lista = []

mixta=["hola", 5, 3.5, True, False, "adios"]

lista1=[x**2 for x in range(10)]
print (lista1)

print(list(range(99,0,-1)))


pares = [x for x in range(100) if x%2==0]

print(pares)


impares = [x for x in range(100) if x%2!=0]

print("impares",impares)

alumnos = ["Steve","Josue1","Josue2","Melissa"]
notas = [89,61,60,95]
tipodesangre= ["O+","O-","A+","B+"]
color= ["rojo","azul","verde","amarillo"]
print(list(zip(alumnos,notas,tipodesangre,color)))

for st in zip(alumnos,notas,tipodesangre,color):
    print(st)       
    

for n in range(10):
    lista.append(n)
print(lista)


def suma(a,b):
    return a+b

print(suma(5,6))

def greet(name:str) -> None:
    print("Hello",name)

greet("Juan")

A=[1,2,3]
B=[4,5,6]

dot = [A[x]*B[x] for x in range(len(A))]
suma=sum(dot)

print(suma)

dot2 =  sum([x*y for(x,y) in zip(A,B)])
print(dot2)

dot3 = sum([x*y for x in A for y in B])
print(dot3)



A = np.array([1,2,3])
B = np.array([4,5,6])

dot4 = np.dot(A,B)
print(dot4)


entradas = [1,2,3,4,5]
pesos_capa=[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]
sesgos_Capa = [2,3,0.5]
salidas= []
for pesos,sesgo in zip(pesos_capa,sesgos_Capa):
    salida_neurona = sum([i*w for i,w in zip(entradas,pesos)])+sesgo
    salidas.append(salida_neurona)

print(salidas) #salidas de la funcion de transferencia



entradas2 = [1,2,3,2.5]
pesos_capa2=[0.2,0.8,-0.5,1],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]
sesgos_Capa2 = [2,3,0.5]
salidas2 = np.dot(pesos_capa2,entradas2)+sesgos_Capa2
print(salidas2) #salidas de la funcion de transferencia


entradasEj=[1,2,9.8]
pesosEj=[0.1,0.5,2.9],[0.7,0.2,0.1]
sesgoEj=[3.3,2]
salidasEj= np.dot(pesosEj,entradasEj)+sesgoEj
print(salidasEj)


np.random.randn(3,4)
print(np.zeros((1,4)))

