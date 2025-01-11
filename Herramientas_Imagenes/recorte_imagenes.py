import cv2
import mediapipe as mp
import os

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Carpeta de entrada y salida
input_folder = r"C:\Users\57322\Downloads\sin_recortar_A"  # Carpeta con imágenes sin recortar
output_folder = r"D:\SEMESTRE7\Dedos\Fotos\Validacion\Letra_A"  # Carpeta para guardar imágenes recortadas

# Crear carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterar sobre todas las imágenes en la carpeta de entrada
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Convertir imagen a RGB (MediaPipe usa RGB)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen con MediaPipe Hands
        result = hands.process(rgb_image)
        
        if result.multi_hand_landmarks:  # Si se detectan manos
            for hand_landmarks in result.multi_hand_landmarks:
                # Obtener el bounding box
                h, w, c = image.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0

                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)

                # Añadir margen al bounding box (opcional)
                margin = 20  # Márgenes generales
                right_margin = 40  # Margen adicional hacia la derecha

                x_min = max(0, x_min - margin)  # Margen normal hacia la izquierda
                y_min = max(0, y_min - margin)  # Margen normal hacia arriba
                x_max = min(w, x_max + margin + right_margin)  # Más margen hacia la derecha
                y_max = min(h, y_max + margin)  # Margen normal hacia abajo


                # Recortar la imagen
                cropped_image = image[y_min:y_max, x_min:x_max]

                # Guardar la imagen recortada
                output_path = os.path.join(output_folder, f"cropped_{filename}")
                cv2.imwrite(output_path, cropped_image)

print("Recorte completado.")
