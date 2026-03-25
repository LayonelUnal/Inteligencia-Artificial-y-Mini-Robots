import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA

# --- Cargar los datos --
iris = load_iris()
X, y = iris.data, iris.target  # X = características (tamaño de pétalos/sépalos), y = tipo de flor

# --- Dividir en entrenamiento y prueba
# 70% para entrenar, 30% para probar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --- Buscar los mejores parámetros con Grid Search ---
# Probamos diferentes combinaciones de parámetros para encontrar la mejor
param_grid = {
    'C': [0.1, 1, 10],              # Controla qué tan estricto es el modelo con los errores
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']      # Controla que tan lejos "mira" cada punto de datos
}

# GridSearchCV prueba todas las combinaciones con validación cruzada (5 partes)
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)  # Entrenamos con todos los parámetros posibles

# El mejor modelo encontrado por la búsqueda
best_svm = grid.best_estimator_
print('Beste Params:', grid.best_params_)  # Muestra los mejores parámetros

# --- Evaluar el modelo ---
y_pred = best_svm.predict(X_test)  # El modelo
print('Accuracy:', accuracy_score(y_test, y_pred))  #
print(classification_report(y_test, y_pred))        # Reporte detallado por clase

# --- Reducir a 2 dimensiones para poder visualizar ---
# Los datos tienen 4 dimensiones, pero solo podemos dibujar 2
# PCA encuentra las 2 direcciones más importantes en los datos
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)  # Transformamos los datos de prueba a 2D

# --- Crear y guardar el gráfico
plt.figure(figsize=(8, 6))
# Dibujamos cada punto con un color según la clase predicha
scatter = plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='viridis')
plt.xlabel('PCA 1')   # Primer componente principal
plt.ylabel('PCA 2')   # Segundo componente principal
plt.title('SVM-Entscheidungsstruktur (PCA-Projektion)')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Klasse')  # Leyenda con los 3 tipos de flores
plt.tight_layout()
plt.savefig('svm_iris_pca.png', dpi=300)  # Guardamos el gráfico como imagen PNG
plt.close()
