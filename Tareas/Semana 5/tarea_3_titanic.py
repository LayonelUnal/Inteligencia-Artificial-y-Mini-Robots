# Ejercicio 5.3 - Titanic (Kaggle) con Keras (MLP)
#Pipeline für einen Kaggle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1) Cargar CSV
df = pd.read_csv("C:\\Users\\tilln\\Documents\\IA Mini Robos\\Titanic-Dataset.csv")

print(df.head())
print(df.columns)


# ['PassengerId','Survived','Pclass','Name','Sex','Age',
#  'SibSp','Parch','Ticket','Fare', ... evtl. Cabin, Embarked]

# 2) Selección de características y objetivo
target_col = "Survived"
feature_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]  # einfache, sinnvolle Features

data = df[feature_cols + [target_col]].copy()

# 3) Tratamiento de valores faltantes
# Age und ggf. Fare können NaNs haben -> mit median füllen
for col in ["Age", "Fare"]:
    if col in data.columns:
        data[col] = data[col].fillna(data[col].median())

# 4) Codificación de variables categóricas
# 'Sex' -> 0/1; wenn 'Embarked' oder andere Kategorische Features genutzt werden, get_dummies()
data["Sex"] = data["Sex"].map({"male": 0, "female": 1}).astype("int32")

# Optional: One-Hot-Encoding für Pclass (statt numerisch)
# data = pd.get_dummies(data, columns=["Pclass"], drop_first=True)

X = data.drop(columns=[target_col])
y = data[target_col].astype("int32")

print("\nX shape:", X.shape)
print("y value counts:\n", y.value_counts())

# 5) Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Escalado de características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

input_dim = X_train_scaled.shape[1]

# 7) Definición del modelo (MLP)
model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),   # salida binaria: probabilidad de sobrevivir
])

model.summary()

# 8) Compilación
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# 9) Entrenamiento
history = model.fit(
    X_train_scaled,
    y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# test
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")

# 11) Métricas
y_pred_prob = model.predict(X_test_scaled, verbose=0).ravel()
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

print("\nMatriz de confusión:")
print(confusion_matrix(y_test, y_pred))

#  predicción
sample = X_test.iloc[:5]
sample_scaled = scaler.transform(sample)
sample_pred_prob = model.predict(sample_scaled, verbose=0).ravel()
sample_pred = (sample_pred_prob >= 0.5).astype(int)

print("\nEjemplos de predicción (primeras 5 filas del test):")
for i in range(len(sample)):
    print(f"Fila {sample.index[i]} -> ProbSurv={sample_pred_prob[i]:.3f}, "
          f"Pred={sample_pred[i]}, Real={y_test.iloc[i]}")
