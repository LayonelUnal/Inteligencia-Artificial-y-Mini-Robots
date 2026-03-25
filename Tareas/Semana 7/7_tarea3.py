import matplotlib
matplotlib.use("Agg")  # Importante: no usamos ventana gráfica, solo guardamos el archivo

import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

# --- Cargar los datos ---
# Breast Cancer: datos reales de tumores — 569 pacientes, 30 características por paciente
# y = 0 significa maligno (peligroso), y = 1 significa benigno (no peligroso)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# --- Dividir en entrenamiento y prueba ---
# 80% para entrenar, 20% para probar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Aprendemos la escala con los datos de entrenamiento
X_test_scaled = scaler.transform(X_test)        # Aplicamos la misma escala a los datos de prueba

# Probamos todos los valores de k entre 1 y 30
k_range = range(1, 31)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # cv=10: dividimos los datos en 10 partes y probamos el modelo 10 veces
    cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=10)
    scores.append(np.mean(cv_scores))  # Guardamos el promedio de las 10 pruebas

# El k con la accuracy más alta es el mejor
best_k = k_range[np.argmax(scores)]
print(f'Bestes k: {best_k}, CV-Accuracy: {scores[np.argmax(scores)]:.3f}')

# --- Entrenar el modelo final con el mejor k ---
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)  # Entrenamos con todos los datos de entrenamiento

# --- Evaluar en los datos de prueba ---
y_pred = knn.predict(X_test_scaled)  # El modelo predice si cada tumor es maligno o benigno
print('Test-Accuracy:', np.mean(y_pred == y_test))  # Porcentaje de predicciones correctas

# Plot
plt.figure()
plt.plot(k_range, scores, marker='o')
plt.xlabel('k')           #
plt.ylabel('CV-Accuracy') # Eje vertical: qué tan bien funciona el modelo
plt.title('KNN: Wahl des besten k')
plt.grid(True)            # Líneas de fondo para leer mejor el gráfico
plt.tight_layout()
plt.savefig('knn_elbow.png', dpi=300)  # Guardamos el gráfico como imagen PNG
plt.close()
