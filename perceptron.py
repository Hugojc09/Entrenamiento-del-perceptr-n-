
import numpy as np
import matplotlib.pyplot as plt

# Datos: edad, peso y altura
animales = ["Perro", "Gato", "Gallina", "Pato", "Caballo", "Aguila"]

X = np.array([
    [5, 20, 50],
    [3, 5, 25],
    [2, 3, 40],
    [1, 4, 35],
    [8, 400, 160],
    [4, 6, 60]
], dtype=float)

# 1 = peso alto (50 kg o mas)
# 0 = peso bajo
d = np.array([0, 0, 0, 0, 1, 0])

# Normalizar datos
X = X / np.max(X, axis=0)

# Pesos iniciales
w = np.zeros(3)
b = 0.0
eta = 0.1

# Guardar errores
lista_errores = []

# Entrenamiento
for epoca in range(20):

    errores = 0
    error_total = 0

    for i in range(len(X)):

        # Combinacion lineal
        z = np.dot(X[i], w) + b

        # Funcion de activacion
        y = 1 if z >= 0 else 0

        # Calcular error
        e = d[i] - y

        error_total += abs(e)

        if e != 0:
            errores += 1

        # Actualizar pesos y sesgo
        w = w + eta * e * X[i]
        b = b + eta * e

    lista_errores.append(error_total)

    print("Epoca:", epoca + 1)
    print("Pesos:", np.round(w, 2))
    print("Sesgo:", round(b, 2))
    print("Error total:", error_total)
    print("Numero de errores:", errores)
    print("--------------------------")

# Clasificacion final
predicciones = []

print("\nRESULTADOS FINALES")

for i in range(len(X)):

    z = np.dot(X[i], w) + b
    y = 1 if z >= 0 else 0

    predicciones.append(y)

    resultado = "Peso alto" if y == 1 else "Peso bajo"

    print(animales[i], ":", resultado)

# GRAFICA 1: Animales
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection="3d")

colores = ["blue" if valor == 0 else "red" for valor in d]

ax.scatter(
    X[:, 0], X[:, 1], X[:, 2],
    c=colores, s=100
)

for i, nombre in enumerate(animales):
    ax.text(X[i, 0], X[i, 1], X[i, 2], nombre)

ax.set_xlabel("Edad")
ax.set_ylabel("Peso")
ax.set_zlabel("Altura")
ax.set_title("Datos de los animales")

plt.show()

# GRAFICA 2: Errores por epoca
plt.plot(range(1, 21), lista_errores, marker="o")
plt.title("Errores durante el entrenamiento")
plt.xlabel("Epocas")
plt.ylabel("Errores")
plt.grid()
plt.show()

# GRAFICA 3: Resultados
posiciones = np.arange(len(animales))
ancho = 0.35

plt.bar(posiciones - ancho/2, d, ancho, label="Esperado")
plt.bar(posiciones + ancho/2, predicciones, ancho, label="Prediccion")

plt.xticks(posiciones, animales)
plt.title("Clasificacion de animales")
plt.ylabel("Resultado")
plt.legend()
plt.grid(axis="y")
plt.show()
