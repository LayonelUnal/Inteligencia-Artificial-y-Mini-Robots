# Aufgabe 1: Zwei kleine Netze (NAND und XOR) mit numpy
# Ejercicio 5.9.1: Dos redes neuronales (NAND y XOR) con numpy

import numpy as np

np.random.seed(0)  # Zufallssamen für Reproduzierbarkeit
# Semilla para -> resultados sean reproducibles

# Aktivierungsfunktionen
# Funciones de activación
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    # Ableitung der Sigmoid-Funktion, ausgedrückt in a = sigmoid(z)
    # Derivada de la sigmoide expresada en términos de a = sigmoid(z)
    return a * (1 - a)

class SimpleMLP:
    """
    Kleine MLP mit zwei Hidden-Layern.
    Pequeña red MLP con dos capas ocultas.
    """
    def __init__(self, input_dim, hidden1, hidden2, output_dim, lr=0.5):
        self.lr = lr
        # Gewichte: kleine Zufallswerte
        # Pesos inicializados con valores aleatorios pequeños
        self.W1 = np.random.randn(hidden1, input_dim) * 0.1
        self.b1 = np.zeros((hidden1, 1))
        self.W2 = np.random.randn(hidden2, hidden1) * 0.1
        self.b2 = np.zeros((hidden2, 1))
        self.W3 = np.random.randn(output_dim, hidden2) * 0.1
        self.b3 = np.zeros((output_dim, 1))

    def forward(self, X):
        """
        Vorwärtspropagation durch das Netz.
        Propagación hacia adelante a través de la red.
        X: (input_dim, m), m = Anzahl der Beispiele
        """
        Z1 = self.W1 @ X + self.b1
        A1 = sigmoid(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = sigmoid(Z2)
        Z3 = self.W3 @ A2 + self.b3
        A3 = sigmoid(Z3)
        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}
        return A3, cache

    def backward(self, Y, cache):
        """
        Backpropagation für quadratischen Fehler:
        Backpropagation para el error cuadrático:
        E = 1/2 ||A3 - Y||^2
        """
        m = Y.shape[1]
        X = cache["X"]
        A1, A2, A3 = cache["A1"], cache["A2"], cache["A3"]

        # Ausgangsschicht
        # Capa de salida
        dA3 = (A3 - Y)              # dE/dA3
        dZ3 = dA3 * sigmoid_deriv(A3)
        dW3 = (dZ3 @ A2.T) / m
        db3 = np.sum(dZ3, axis=1, keepdims=True) / m

        # Zweite Hidden-Schicht
        # Segunda capa oculta
        dA2 = self.W3.T @ dZ3
        dZ2 = dA2 * sigmoid_deriv(A2)
        dW2 = (dZ2 @ A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Erste Hidden-Schicht
        # Primera capa oculta
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * sigmoid_deriv(A1)
        dW1 = (dZ1 @ X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Gradient Descent Update der Gewichte und Biases
        # Actualización de pesos y sesgos con descenso del gradiente
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, Y, epochs=5000, print_every=1000):
        """
        Training über mehrere Epochen.
        Entrenamiento de la red durante varias épocas.
        """
        for i in range(epochs):
            A3, cache = self.forward(X)
            loss = np.mean(0.5 * (A3 - Y) ** 2)  # quadratischer Fehler
            # error cuadrático medio
            self.backward(Y, cache)
            if (i + 1) % print_every == 0:
                print(f"Epoche {i+1}, Loss: {loss:.6f}")

    def predict(self, X):
        """
        Binäre Vorhersage (0/1) mit Schwelle 0.5.
        Predicción binaria (0/1) usando umbral 0.5.
        """
        A3, _ = self.forward(X)
        return (A3 > 0.5).astype(int)

# Trainingsdaten für logische Gatter (Wahrheitstabelle)
# Datos de entrenamiento (tabla de verdad) para las compuertas lógicas
X_logic = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])   # Form (2,4), jede Spalte = (x1, x2)
     # Forma (2,4), cada columna = (x1, x2)

# NAND-Labels
# Salidas deseadas para la compuerta NAND
Y_nand = np.array([[1, 1, 1, 0]])  # Form (1,4)

# XOR-Labels
# Salidas deseadas para la compuerta XOR
Y_xor = np.array([[0, 1, 1, 0]])   # Form (1,4)

X_train = X_logic


print("=== Red neuronal para compuerta NAND ===")
nand_net = SimpleMLP(input_dim=2, hidden1=4, hidden2=4, output_dim=1, lr=0.5)
nand_net.fit(X_train, Y_nand, epochs=5000, print_every=1000)
print("Entradas:")
print(X_train)
print("Salidas predichas (redondeadas):")
print(nand_net.predict(X_train))

#print("\n=== Netz für XOR ===")
print("=== Red neuronal para compuerta XOR ===")
xor_net = SimpleMLP(input_dim=2, hidden1=4, hidden2=4, output_dim=1, lr=0.5)
xor_net.fit(X_train, Y_xor, epochs=10000, print_every=2000)
print("Entradas:")
print(X_train)
print(" Salidas predichas (redondeadas):")
print(xor_net.predict(X_train))
