# Ejercicio 5.2 - Fashion-MNIST con TensorFlow/Keras (CNN)



import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Para reproducibilidad
np.random.seed(0)
tf.random.set_seed(0)
random.seed(0)

# 1) Carga y preprocesado
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
               "Sandal","Shirt","Sneaker","Bag","Ankle boot"]

# Normalización y añadir canal (N,28,28,1)
x_train = (x_train / 255.0).astype("float32")[..., None]
x_test  = (x_test  / 255.0).astype("float32")[..., None]

print("Forma de x_train:", x_train.shape)
print("Forma de x_test :", x_test.shape)

# CNN
model = keras.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])

model.summary()

# 3) Compilación
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# EarlyStopping ( evitar overfitting)
callbacks = [
    keras.callbacks.EarlyStopping(
        patience=3,
        monitor="val_accuracy",
        restore_best_weights=True
    )
]

# 4) Training/Entrenamiento
hist = model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=128,
    callbacks=callbacks
)

# test
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f} | loss: {test_loss:.4f}")

# 6) Métricas detalladas
y_pred_probs = model.predict(x_test, verbose=0)
y_pred = y_pred_probs.argmax(axis=1)

print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=class_names))

######
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de confusión:\n", cm)

# 7) Muestra aleatoria de predicciones
idxs = np.random.choice(len(x_test), 9, replace=False)
plt.figure(figsize=(6,6))
for i, idx in enumerate(idxs, 1):
    plt.subplot(3,3,i)
    plt.imshow(x_test[idx].squeeze(), cmap="gray")
    true_label = class_names[y_test[idx]]
    pred_label = class_names[y_pred[idx]]
    plt.title(f"y:{true_label}\nŷ:{pred_label}")
    plt.axis("off")
plt.tight_layout()
plt.show()
