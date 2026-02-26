import math
import random
import numpy as np

# --- PARAMETER (kannst du bei Bedarf anpassen) ---
longCrom = 100  # Länge des Chromosoms (höher = genauere Lösung)
K = 50  # Populationsgröße
M = 100  # Anzahl der Generationen
pm = 0.05  # Mutationswahrscheinlichkeit



# ==========================================
def ecuacion(x):
    valor = math.sin(10 * math.pi * x) + 1
    return valor


# ==========================================
#  INTERVALL 3.10.1: x ∈ [0,1]
# ==========================================
def decodificar(cromosoma):
    xi = 0.0  # x_min = 0
    xf = 1.0  # x_max = 1

    Max = float(2 ** longCrom)
    valorDecimal = 0.0
    for i in range(longCrom):
        if cromosoma[i] == 1:
            valorDecimal += 2.0 ** (longCrom - i - 1)

    valDeco = ((xf - xi) / Max) * valorDecimal + xi
    return valDeco


def genera(K, longCrom):
    Pob_nueva = np.zeros([K, longCrom], dtype=int)
    j = 0
    while j < K:
        cromosoma = [random.randint(0, 1) for i in range(longCrom)]
        Pob_nueva[j] = cromosoma
        j += 1
    return Pob_nueva


def evalua(Pob_nueva):
    vectorX = np.zeros(K, dtype=float)
    aptitud = np.zeros(K, dtype=float)

    i = 0
    while i < K:
        t, x = res_Funcion(Pob_nueva[i])
        vectorX[i] = x

        # FITNESS FÜR MAXIMIERUNG von sin(10πx) + 1
        # Da der Wertebereich [0, 2] ist, addieren wir +1000 für positive Fitness:
        aptitud[i] = t + 1000

        i += 1

    Apt_total = float(sum(aptitud))
    if Apt_total == 0:
        Apt_total = 1e-9

    probab = [j / Apt_total for j in aptitud]
    probab = np.array(probab)

    maxIndex = np.argmax(probab)
    probab[maxIndex] = 0.99

    return probab, vectorX


def res_Funcion(cromosoma):
    x = decodificar(cromosoma)
    funcion = ecuacion(x)
    return funcion, x


def cruce(Pob_nueva, Probabilidad):
    maxIndex = np.argmax(Probabilidad)
    i = 0
    while (i < K - 1):
        if Probabilidad[i] < 0.97:
            rand = random.randint(2, longCrom - 1)
            padre1 = Pob_nueva[i].copy()
            padre2 = Pob_nueva[i + 1].copy()
            j = rand
            while (j < longCrom):
                bit = padre1[j]
                padre1[j] = padre2[j]
                padre2[j] = bit
                j += 1
            Pob_nueva[i] = padre1
            Pob_nueva[i + 1] = padre2
        else:
            if maxIndex % 2 == 0:
                Pob_nueva[i] = Pob_nueva[maxIndex].copy()
            else:
                Pob_nueva[i + 1] = Pob_nueva[maxIndex].copy()
        i += 2
    return Pob_nueva


def muta(Pob_nueva, pm):
    totalbits = K * longCrom
    segmento = 1 / pm
    n_segmentos = int(totalbits / segmento)
    i = 0
    while (i < n_segmentos - 1):
        aleatorio = random.randint(0, int(segmento) - 1)
        posic = int(i * segmento + aleatorio)
        y = int(posic / longCrom)
        cromosoma = Pob_nueva[y]
        x = posic - longCrom * y
        if (cromosoma[x - 1] == 0):
            cromosoma[x - 1] = 1
        else:
            cromosoma[x - 1] = 0
        i += 1
    return Pob_nueva


def seleccion_ruleta(poblacion, probabilidad):
    chosen = []
    while len(chosen) < K:
        for n in range(K):
            r = random.random()
            for i, individuo in enumerate(poblacion):
                if i >= K:
                    break
                if r <= probabilidad[i]:
                    chosen.append(list(individuo))
                    break
            if len(chosen) >= K:
                break
    return chosen


# ==========================================
# Resultados: TAREA 3.10.1
# ==========================================
print("Algoritmo Genético 3.10.1:")
print("f(x) = sin(10πx) + 1, x ∈ [0,1]")
print("Objetivo: El máximo global\n")

Pob_nueva3 = genera(K, longCrom)
prob_cromosoma, vectorX = evalua(Pob_nueva3)
i = 0

while (i < M):
    Pob_vieja = Pob_nueva3
    Pob_nueva1 = seleccion_ruleta(Pob_vieja, prob_cromosoma)
    Pob_nueva2 = cruce(np.array(Pob_nueva1), prob_cromosoma)
    Pob_nueva3 = muta(Pob_nueva2, pm)
    prob_cromosoma, vectorX = evalua(Pob_nueva3)

    maxIndex = np.argmax(prob_cromosoma)
    mejorx = vectorX[maxIndex]
    val = ecuacion(mejorx)

    # Salida cada 10 generaciones
    if i % 10 == 0 or i == M - 1:
        print(f"Generación {i:3d}: Mejor x = {mejorx:.6f} | f(x) = {val:.6f}")

    i += 1

print("\n" + "=" * 60)
print("¡TAREA 3.10.1!")
print(f"El máximo encontrado está en x = {mejorx:.6f}")
print(f"Valor de la función f(x) = {val:.6f}")
print("Teóricamente esperados: x = 0.1, 0.3, 0.5, 0.7, 0.9 (períodos de sin(10πx))")
print("=" * 60)
