# 🤖 Control de Robot Autónomo por Voz: Hacia la IA Física

Este proyecto presenta una simulación interactiva en tiempo real donde un agente (robot) responde a comandos de voz. Más allá de un simple script, esta arquitectura demuestra el manejo de procesos concurrentes y modelado espacial, sentando las bases lógicas para el desarrollo de sistemas ciberfísicos y de automatización.

## 🌍 ¿Cómo se encamina a la IA Física (Physical AI)?
La Inteligencia Artificial Física busca sacar a los algoritmos de los servidores y darles la capacidad de interactuar con el entorno real. Esta simulación actúa como un "gemelo digital" en su etapa más fundamental. 

El flujo de información empleado aquí (Percepción del entorno a través de un micrófono ➔ Procesamiento de Lenguaje Natural ➔ Actuación física en un plano cartesiano) utiliza exactamente la misma arquitectura de toma de decisiones que un vehículo autónomo o un brazo robótico. Abstraemos los componentes mecánicos mediante trigonometría, permitiendo validar la lógica de control antes de llevarla al hardware real.

## 🛠️ Arquitectura y Librerías Utilizadas
Para lograr que el robot "escuche" el entorno y "actúe" visualmente al mismo tiempo sin que el sistema se congele, se diseñó una arquitectura concurrente utilizando:

* **`speech_recognition` (El Oído):** Captura el audio, calibra automáticamente el ruido de fondo y utiliza la API para transcribir la voz a texto en tiempo real.
* **`matplotlib` (El Entorno UI/UX):** A través de su modo interactivo (`plt.ion()`), construimos la interfaz visual que renderiza la posición del robot dinámicamente, garantizando una respuesta visual fluida.
* **`threading` (Procesamiento Paralelo):** Permite aislar el motor de reconocimiento de voz en un hilo independiente (en segundo plano). Así, la interfaz gráfica sigue procesando animaciones mientras el sistema espera el próximo comando de voz.
* **`queue` (Comunicación Segura):** Actúa como el puente o "médula espinal" entre el hilo de voz y el hilo principal de la interfaz. Los comandos de voz se enfilan aquí para que el robot los consuma y ejecute de forma estructurada.

## ⚙️ Dinámica de la Simulación
El modelo del robot mantiene un estado interno basado en coordenadas de posición `(x, y)` y un ángulo de orientación `(theta)`. Al recibir instrucciones específicas ("adelante", "atrás", "izquierda", "derecha"), el sistema actualiza su vector de dirección aplicando funciones trigonométricas básicas (seno y coseno) y redibuja instantáneamente su nueva huella en la cuadrícula.

---

## 🚀 Instrucciones de Ejecución

### 1. Dependencias del Hardware
Se necesita instalar la siguiente dependencia para que Python pueda interactuar con el micrófono de tu equipo:

```bash
pip install pyaudio