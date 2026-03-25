import matplotlib
matplotlib.use("Agg")  #

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Cargar los datos --
# Wine: datos reales de 178 vinos de Italia
wine = load_wine()
X, y = wine.data, wine.target

# --- Dividir en entrenamiento y prueba ---
# 70% para entrenar, 30% para probar
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

#o solo se divide si tiene al menos 10 ejemplos
tree = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    random_state=42
)
tree.fit(X_train, y_train)  # El árbol aprende las reglas con los datos de entrenamiento

# --- Evaluar el modelo ---
y_pred = tree.predict(X_test)  # El árbol predice el tipo de vino para cada ejemplo
print('Accuracy:', accuracy_score(y_test, y_pred))  # Porcentaje de predicciones correctas

# Valor entre 0 y 1 — más alto significa más importante
print('Feature Importance:', tree.feature_importances_)

# --- Dibujar el árbol y guardar como PNG ---
# plot_tree dibuja todas las reglas del árbol de forma visual
plt.figure(figsize=(16, 10))  # Figura grande para que el árbol sea fácil de leer
plot_tree(
    tree,
    feature_names=wine.feature_names,  # Nombres de las 13 características químicas
    class_names=wine.target_names,     # Nombres de los 3 tipos de vino
    filled=True,    # Colores diferentes para cada clase
    rounded=True,   # Bordes redondeados para los nodos
    fontsize=8      # Texto pequeño para que todo quepa bien
)
plt.tight_layout()
plt.savefig("decision_tree_wine.png", dpi=300)  # Guardamos el árbol como imagen PNG
plt.close()  #