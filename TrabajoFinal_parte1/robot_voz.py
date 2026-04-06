import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
import threading
import queue
import sys

# =================================================================
# 1. COLA DE MENSAJES (Comunicación segura entre hilos)
# =================================================================
cola_comandos = queue.Queue()

# =================================================================
# 2. CAPA DE ABSTRACCIÓN DE HARDWARE (Robot Simulado)
# =================================================================
class RobotSimulado:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0 
        self.step = 1.0 

    def mover(self, comando):
        if comando == "adelante":
            self.x += self.step * np.cos(self.theta)
            self.y += self.step * np.sin(self.theta)
        elif comando == "atras":
            self.x -= self.step * np.cos(self.theta)
            self.y -= self.step * np.sin(self.theta)
        elif comando == "izquierda":
            self.theta += np.pi / 4
        elif comando == "derecha":
            self.theta -= np.pi / 4

# =================================================================
# 3. MOTOR DE RECONOCIMIENTO DE VOZ (Hilo Independiente)
# =================================================================
def escuchar_voz():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        cola_comandos.put(("ESTADO", "Calibrando ruido... silencio por favor."))
        recognizer.adjust_for_ambient_noise(source, duration=2)
        cola_comandos.put(("ESTADO", "SISTEMA LISTO. Di 'Adelante', 'Atrás', 'Izquierda', 'Derecha' o 'Salir'"))
        
        while True:
            try:
                # Escucha con límite de tiempo para no quedarse bloqueado eternamente
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)
                
                # Para un sistema robusto sin internet, aquí se usaría un modelo local (Vosk/Whisper)
                texto = recognizer.recognize_google(audio, language="es-ES").lower()
                print(f"🎙️ Transcripción: '{texto}'")
                
                if "adelante" in texto: cola_comandos.put(("COMANDO", "adelante"))
                elif "atrás" in texto or "atras" in texto: cola_comandos.put(("COMANDO", "atras"))
                elif "izquierda" in texto: cola_comandos.put(("COMANDO", "izquierda"))
                elif "derecha" in texto: cola_comandos.put(("COMANDO", "derecha"))
                elif "salir" in texto: 
                    cola_comandos.put(("COMANDO", "salir"))
                    break # Rompe el bucle de voz
                else:
                    cola_comandos.put(("ESTADO", f"Comando ignorado: '{texto}'"))
                    
            except sr.WaitTimeoutError:
                # Si nadie habla en 3 segundos, simplemente vuelve a escuchar
                continue 
            except sr.UnknownValueError:
                cola_comandos.put(("ESTADO", "⚠️ No pude entender, repite..."))
            except sr.RequestError:
                cola_comandos.put(("ESTADO", "❌ Error de conexión con la API."))

# =================================================================
# 4. BUCLE PRINCIPAL INTERACTIVO
# =================================================================
if __name__ == "__main__":
    robot = RobotSimulado()
    
    # Activa el modo interactivo de Matplotlib
    plt.ion() 
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Inicia el hilo de voz
    hilo_voz = threading.Thread(target=escuchar_voz)
    hilo_voz.daemon = True # El hilo muere si el programa principal se cierra
    hilo_voz.start()
    
    estado_actual = "Iniciando sistema..."
    
    # El bucle mantiene el programa vivo mientras la ventana esté abierta
    while plt.fignum_exists(fig.number):
        # 1. Revisar si hay mensajes nuevos del hilo de voz
        try:
            tipo_msj, valor = cola_comandos.get_nowait()
            if tipo_msj == "COMANDO":
                if valor == "salir":
                    print("Cerrando el sistema por comando de voz...")
                    break
                robot.mover(valor)
                estado_actual = f"Ejecutando: Mover {valor}"
            elif tipo_msj == "ESTADO":
                estado_actual = valor
        except queue.Empty:
            pass # No hay comandos nuevos, continuamos
            
        # 2. Limpiar y redibujar el gráfico (esto da la interactividad fluida)
        ax.clear()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_title("Robot Autónomo - Control por Voz en Tiempo Real")
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Dibujar el robot (cuerpo azul, frente rojo)
        ax.plot(robot.x, robot.y, 'bo', markersize=15)
        ax.plot([robot.x, robot.x + np.cos(robot.theta)], 
                [robot.y, robot.y + np.sin(robot.theta)], 'r-', lw=3)
        
        # Dibujar el texto de estado en la parte superior del gráfico
        ax.text(-9.5, 9, estado_actual, fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        # 3. Pausar brevemente para que Matplotlib procese eventos (clave para no cerrarse ni congelarse)
        plt.pause(0.1)


   # Si salimos del bucle (ventana cerrada o comando "salir")
    plt.ioff() # Apaga el modo interactivo para evitar cuelgues
    plt.close('all')
    print("🤖 Sistema apagado correctamente.")