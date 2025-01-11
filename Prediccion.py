import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Rutas de los archivos
modelo = r'D:\SEMESTRE7\Deteccion-y-Clasificacion-de-Manos\ModeloVocales.keras'
peso = r'D:\SEMESTRE7\Deteccion-y-Clasificacion-de-Manos\pesosVocales.weights.h5'

# Cargamos el modelo y los pesos
cnn = load_model(modelo)  # Cargamos el modelo (incluye los pesos integrados)
cnn.load_weights(peso)  # Cargamos los pesos si están en un archivo separado

# Ruta de las imágenes de validación
direccion = r'D:\SEMESTRE7\Dedos\Fotos\Validacion'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

# Leemos la cámara
cap = cv2.VideoCapture(0)

# Creamos un objeto para almacenar la detección y el seguimiento de las manos
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()  # Parámetros predeterminados para la detección de manos

# Método para dibujar las manos
dibujo = mp.solutions.drawing_utils  # Dibujamos los 21 puntos clave de la mano

while True:
    ret, frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []  # Coordenadas de los puntos clave de la mano

    if resultado.multi_hand_landmarks:  # Si se detectan manos
        for mano in resultado.multi_hand_landmarks:
            for id, lm in enumerate(mano.landmark):
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x * ancho), int(lm.y * alto)
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

            if posiciones:
                pto_i5 = posiciones[9]
                x1, y1 = (pto_i5[1] - 80), (pto_i5[2] - 80)
                ancho, alto = (x1 + 80), (y1 + 80)
                x2, y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                x = img_to_array(dedos_reg)
                x = np.expand_dims(x, axis=0)
                vector = cnn.predict(x)
                resultado = vector[0]
                respuesta = np.argmax(resultado)

                if respuesta == 1:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[0]), (x1, y1 - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
                elif respuesta == 0:
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[1]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("Video", frame)
    k = cv2.waitKey(1)
    if k == 27:  # Presiona 'Esc' para salir
        break

cap.release()
cv2.destroyAllWindows()