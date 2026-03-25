import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

# --- Crear los datos ---
# Generamos 50.000 ejemplos de multiplicaciones de matrices 2x2
n_samples = 50000
A_list, B_list, C_list = [], [], []
np.random.seed(42)  # Para que los números aleatorios sean siempre iguales

for _ in range(n_samples):
    A = np.random.randint(-20, 21, size=(2, 2))  # Matriz A con números entre -20 y 20
    B = np.random.randint(-20, 21, size=(2, 2))  # Matriz B con números entre -20 y 20
    C = np.dot(A, B)  # Resultado correcto (matemática normal)
    A_list.append(A.flatten())  # Convertimos la matriz a una lista de 4 números
    B_list.append(B.flatten())
    C_list.append(C.flatten())

# Juntamos A y B como entrada (8 números por ejemplo)
X = np.hstack((A_list, B_list))  # Forma: (50000, 8)
# El resultado C tiene 4 números (los 4 elementos de la matriz)
y = np.array(C_list)  # Forma: (50000, 4)

# --- Dividir en entrenamiento y prueba ---
# 80% para entrenar, 20% para probar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Entrenar
models = []
for i in range(4):
    model = LinearRegression()  # Modelo de regresión lineal simple
    model.fit(X_train, y_train[:, i])  # El modelo aprende con los datos de entrenamiento
    models.append(model)

# --- Ver qué tan bien funciona el modelo ---
# Predecimos los resultados para los datos de prueba
y_pred_test = np.array([model.predict(X_test) for model in models]).T
mse = mean_squared_error(y_test, y_pred_test)
print(f'MSE on test set: {mse}')  # Error promedio — queremos que sea muy pequeño

# --- Comparar 10 ejemplos: matemática vs. modelo ---
test_Cs_analytical = []
test_Cs_ml = []

for _ in range(10):
    A_test = np.random.randint(-20, 21, size=(2, 2))
    B_test = np.random.randint(-20, 21, size=(2, 2))

    # Resultado real con multiplicación normal
    C_anal = np.dot(A_test, B_test).flatten()
    test_Cs_analytical.append(C_anal)

    # Resultado del modelo de machine learning
    X_single = np.hstack((A_test.flatten(), B_test.flatten())).reshape(1, -1)
    C_ml = [model.predict(X_single)[0] for model in models]
    test_Cs_ml.append(C_ml)

print('Analytical C:\n', np.array(test_Cs_analytical))
print('ML predictions:\n', np.array(test_Cs_ml))
print('Differences:\n', np.abs(np.array(test_Cs_analytical) - np.array(test_Cs_ml)))  # Diferencia absoluta


# --- Medir el tiempo de cada método ---
# ¿Cuánto tiempo necesita la multiplicación normal?
def analytical_time(n):
    start = time.time()
    for _ in range(n):
        np.dot(np.random.randint(-20, 21, (2, 2)), np.random.randint(-20, 21, (2, 2)))
    return time.time() - start


# ¿Cuánto tiempo necesita el modelo de ML?
def ml_time(n, models):
    # Primero creamos todos los datos (esto no contamos en el tiempo)
    Xs = [np.hstack((np.random.randint(-20, 21, (2, 2)).flatten(),
                     np.random.randint(-20, 21, (2, 2)).flatten())) for _ in range(n)]
    Xs = np.array(Xs)
    start = time.time()
    for i in range(4):
        models[i].predict(Xs)  # El modelo hace n predicciones a la vez
    return time.time() - start


# Comparamos con 10.000 operaciones
n_trials = 10000
t_anal = analytical_time(n_trials)
t_ml = ml_time(n_trials, models)
print(f'Time analytical: {t_anal:.4f}s, ML: {t_ml:.4f}s (ML {t_ml / t_anal:.2f}x slower)')
