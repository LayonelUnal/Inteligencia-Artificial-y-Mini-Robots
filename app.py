import random

# ============================================
# 1. PARAMETER UND INITIALISIERUNG
# ============================================
random.seed(42)  # para reproducir

N_PARTIDOS = 5
N_CURULES = 50
N_ENTIDADES = 50

# Namen der Parteien
partidos = ["Partido A", "Partido B", "Partido C", "Partido D", "Partido E"]

# Asientos:
# Ejemplo: [18, 5, 2, 15, 10] (Summe = 50)
curules = [18, 5, 2, 15, 10]
prop_curules = [c / N_CURULES for c in curules]  # Ziel-Prozente (z.B. 36%, 10%, 4%, 30%, 20%)

# 50 Behörden mit zufälligen Gewichten (Macht) von 1 bis 100
pesos = [random.randint(1, 100) for _ in range(N_ENTIDADES)]
poder_total = sum(pesos)

# AG Parameter
POP_SIZE = 100
GENERACIONES = 400
TASA_MUTACION = 0.05
TAM_TORNEO = 3


# ============================================
# 2. Funcion
# ============================================

# Ein Chromosom ist eine Liste von 50 Zahlen (0 bis 4).
# z.B. [0, 4, 1, 0, ...] bedeutet: Entität 0 geht an Partei 0 (A), Entität 1 an Partei 4 (E) usw.
def crear_individuo():
    return [random.randint(0, N_PARTIDOS - 1) for _ in range(N_ENTIDADES)]


# Wie gut ist die Verteilung? (Ziel: Anteil an Macht soll Anteil an Sitzen entsprechen)
def evaluar_fitness(individuo):
    poder_partidos = [0] * N_PARTIDOS

    # 1. Berechne, wie viel Macht jede Partei in diesem Chromosom bekommt
    for idx_entidad, partido in enumerate(individuo):
        poder_partidos[partido] += pesos[idx_entidad]

    # 2. Berechne den Fehler (Mean Squared Error) im Vergleich zu den Sitzen
    error = 0
    for p in range(N_PARTIDOS):
        prop_poder_actual = poder_partidos[p] / poder_total
        error += (prop_poder_actual - prop_curules[p]) ** 2

    # Fitness maximieren: Je kleiner der Fehler, desto höher die Fitness
    return 1.0 / (1.0 + error * 10000)


def seleccion_torneo(poblacion, fitness_list):
    # Wähle zufällig TAM_TORNEO Individuen aus und nimm das beste
    indices = random.sample(range(POP_SIZE), TAM_TORNEO)
    mejor_idx = max(indices, key=lambda i: fitness_list[i])
    return poblacion[mejor_idx][:]


def cruzamiento(p1, p2):
    # 2-Punkt-Crossover (Tausche den Mittelteil zwischen zwei Eltern)
    punto1, punto2 = sorted(random.sample(range(N_ENTIDADES), 2))
    hijo = p1[:]
    hijo[punto1:punto2] = p2[punto1:punto2]
    return hijo


def mutacion(individuo):
    # Ändere zufällig mit 5% Wahrscheinlichkeit die Zuordnung eines Ministeriums
    for i in range(N_ENTIDADES):
        if random.random() < TASA_MUTACION:
            individuo[i] = random.randint(0, N_PARTIDOS - 1)
    return individuo


# ============================================
# 3. Main
# ============================================
print(f"Buscando distribución para {poder_total} puntos de poder en total...\n")

poblacion = [crear_individuo() for _ in range(POP_SIZE)]
mejor_individuo = None
mejor_fitness = -1

for gen in range(GENERACIONES):
    fitness_list = [evaluar_fitness(ind) for ind in poblacion]

    # Bestes Individuum der aktuellen Generation finden
    idx_mejor_actual = fitness_list.index(max(fitness_list))
    if fitness_list[idx_mejor_actual] > mejor_fitness:
        mejor_fitness = fitness_list[idx_mejor_actual]
        mejor_individuo = poblacion[idx_mejor_actual][:]

    # Elitismus: Das beste Individuum kommt direkt in die nächste Generation
    nueva_poblacion = [poblacion[idx_mejor_actual][:]]

    # Rest der Population durch Evolution auffüllen
    while len(nueva_poblacion) < POP_SIZE:
        padre1 = seleccion_torneo(poblacion, fitness_list)
        padre2 = seleccion_torneo(poblacion, fitness_list)
        hijo = cruzamiento(padre1, padre2)
        hijo = mutacion(hijo)
        nueva_poblacion.append(hijo)

    poblacion = nueva_poblacion

# ============================================
# 4. AUSWERTUNG/Resultados
# ============================================
poder_final = [0] * N_PARTIDOS
for idx_entidad, partido in enumerate(mejor_individuo):
    poder_final[partido] += pesos[idx_entidad]

print(f"{'Partido':<12} | {'Curules':>7} | {'% Objetivo':>10} | {'Poder Asignado':>15} | {'% Logrado':>10}")
print("-" * 65)
for p in range(N_PARTIDOS):
    prop_lograda = poder_final[p] / poder_total * 100
    prop_objetivo = prop_curules[p] * 100
    print(
        f"{partidos[p]:<12} | {curules[p]:>7} | {prop_objetivo:>9.1f}% | {poder_final[p]:>15} | {prop_lograda:>9.1f}%")

print("\nMATRIZ DE PODER (Todas las 50 entidades):")
print(f"{'Entidad':<12} | {'Peso':>4} | {'A':>3} | {'B':>3} | {'C':>3} | {'D':>3} | {'E':>3} |")
print("-" * 51)
for i in range(N_ENTIDADES):
    asignacion = ["  -" for _ in range(N_PARTIDOS)]
    partido_asignado = mejor_individuo[i]
    asignacion[partido_asignado] = f"{pesos[i]:>3}"
    print(f"Entidad {i+1:>2}   | {pesos[i]:>4} |", " | ".join(asignacion) + " |")

print("-" * 51)
# Extra: Die Summe ganz unten zum Überprüfen
sumas_str = [f"{poder_final[p]:>3}" for p in range(N_PARTIDOS)]
print(f"{'TOTAL PODER':<12} | {poder_total:>4} |", " | ".join(sumas_str) + " |")
