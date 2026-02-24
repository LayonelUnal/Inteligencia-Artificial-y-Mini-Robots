import random
import numpy as np

# parametros del AG
longCrom = 16  # 16 rutas (4 plantas x 4 ciudades)
K = 100  # tamaño de la poblacion
M = 200  # cuantas generaciones corre
pm = 0.05  # prob de mutacion

# ---- Daten/ datos del problema ----
# Kraftwerke/ plantas: Cali=0, Bogota=1, Medellin=2, Barranquilla=3
plantas_nombres = ["Cali", "Bogotá", "Medellín", "Barranquilla"]
oferta = np.array([3, 6, 5, 4])  # GW que genera cada planta
demanda = np.array([4, 3, 5, 3])  # GW que necesita cada ciudad

# tabla de costos de transporte (de la tarea)
costo_transporte = np.array([
    [1, 4, 3, 6],
    [4, 1, 4, 5],
    [3, 4, 1, 4],
    [6, 5, 4, 1]
])

# costo de generar 1 KW-H por planta
costo_generacion = np.array([680, 720, 660, 750])


# ---- decodificacion ----
# el cromosoma son prioridades entre 0 y 1
# ordenamos de mayor a menor y asignamos GW en ese orden
def decodificar(cromosoma):
    indices_prioridad = np.argsort(cromosoma)[::-1]

    despacho = np.zeros((4, 4), dtype=int)
    of_disp = oferta.copy()
    dem_disp = demanda.copy()

    for idx in indices_prioridad:
        i = idx // 4  # planta
        j = idx % 4  # ciudad destino

        # asignamos lo que se pueda sin pasarse
        asignar = min(of_disp[i], dem_disp[j])
        despacho[i, j] = asignar
        of_disp[i] -= asignar
        dem_disp[j] -= asignar

    return despacho


def res_Funcion(despacho):
    costo_total = 0
    for i in range(4):
        for j in range(4):
            if despacho[i, j] > 0:
                # costo de esta ruta = energia * (transporte + generacion)
                costo_ruta = despacho[i, j] * (costo_transporte[i, j] + costo_generacion[i])
                costo_total += costo_ruta
    return costo_total


def evalua(Pob_nueva):
    aptitud = np.zeros(K, dtype=float)
    costos = np.zeros(K, dtype=float)

    for i in range(K):
        despacho = decodificar(Pob_nueva[i])
        costo = res_Funcion(despacho)
        costos[i] = costo

        # queremos minimizar el costo, entonces la aptitud es la inversa
        # el +1 es para no dividir por cero si el costo fuera 0
        aptitud[i] = 1000000.0 / (costo + 1)

    Apt_total = float(np.sum(aptitud))
    if Apt_total == 0: Apt_total = 1e-9

    probab = aptitud / Apt_total

    # al mejor le ponemos 0.99 para que siempre pase a la siguiente generacion
    maxIndex = np.argmax(probab)
    probab[maxIndex] = 0.99

    return probab, costos


# ---- genera poblacion inicial ----
def genera(K, longCrom):
    # cada gen es un numero entre 0 y 1 que representa la prioridad de esa ruta
    return np.random.rand(K, longCrom)


def cruce(Pob_nueva, Probabilidad):
    maxIndex = np.argmax(Probabilidad)
    i = 0
    Pob_cruza = np.copy(Pob_nueva)
    while (i < K - 1):
        if Probabilidad[i] < 0.97:
            rand = random.randint(1, longCrom - 2)
            padre1 = Pob_cruza[i].copy()
            padre2 = Pob_cruza[i + 1].copy()
            Pob_cruza[i] = np.concatenate((padre1[:rand], padre2[rand:]))
            Pob_cruza[i + 1] = np.concatenate((padre2[:rand], padre1[rand:]))
        else:
            # si uno de los padres es el mejor, lo copiamos directo
            if maxIndex % 2 == 0:
                Pob_cruza[i] = Pob_cruza[maxIndex].copy()
            else:
                Pob_cruza[i + 1] = Pob_cruza[maxIndex].copy()
        i += 2
    return Pob_cruza


def muta(Pob_nueva, pm):
    for i in range(K):
        for j in range(longCrom):
            if random.random() < pm:
                # cambiamos la prioridad de esta ruta por una nueva aleatoria
                Pob_nueva[i][j] = random.random()
    return Pob_nueva


def seleccion_ruleta(poblacion, probabilidad):
    chosen = []
    while len(chosen) < K:
        for n in range(K):
            r = random.random()
            for i, individuo in enumerate(poblacion):
                if i >= K: break
                if r <= probabilidad[i]:
                    chosen.append(individuo.copy())
                    break
            if len(chosen) >= K: break
    return np.array(chosen)


# ---- bucle principal ----
Pob_nueva3 = genera(K, longCrom)
prob_cromosoma, vectorCostos = evalua(Pob_nueva3)

mejor_cromosoma_global = None
menor_costo_global = float('inf')

print("Corriendo AG para despacho de energia...\n")

for i in range(M):
    Pob_vieja = Pob_nueva3
    Pob_nueva1 = seleccion_ruleta(Pob_vieja, prob_cromosoma)
    Pob_nueva2 = cruce(Pob_nueva1, prob_cromosoma)
    Pob_nueva3 = muta(Pob_nueva2, pm)

    prob_cromosoma, vectorCostos = evalua(Pob_nueva3)
    maxIndex = np.argmax(prob_cromosoma)
    costo_actual = vectorCostos[maxIndex]

    # guardamos el mejor de todas las generaciones
    if costo_actual < menor_costo_global:
        menor_costo_global = costo_actual
        mejor_cromosoma_global = Pob_nueva3[maxIndex].copy()

    if i % 20 == 0 or i == M - 1:
        print(f"Generacion {i:3d}: Costo minimo encontrado = ${costo_actual:,.2f}")

# ---- resultados ----
mejor_despacho = decodificar(mejor_cromosoma_global)

print("\n" + "=" * 55)
print("RESULTADO: MEJOR DESPACHO DE ENERGIA")
print("=" * 55)
print(f"Costo total optimizado: ${menor_costo_global:,.2f}\n")

print(f"{'Planta origen':<15} | {'Ciudad destino':<15} | GW enviados")
print("-" * 50)
for i in range(4):
    for j in range(4):
        if mejor_despacho[i, j] > 0:
            print(f"{plantas_nombres[i]:<15} -> {plantas_nombres[j]:<15} | {mejor_despacho[i, j]} GW")

# verificamos que la demanda queda cubierta
print("\nVerificacion de demanda:")
for j in range(4):
    recibido = sum(mejor_despacho[:, j])
    estado = "OK" if recibido == demanda[j] else "FALTA ENERGIA"
    print(f"  {plantas_nombres[j]}: necesita {demanda[j]} GW, recibe {recibido} GW -> {estado}")
