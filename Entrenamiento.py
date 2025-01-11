#------------------------------- Importamos librerías ---------------------------------
import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Preprocesamiento de imágenes
from keras.models import Sequential  # Crear redes neuronales secuenciales
from keras.layers import Dropout, Flatten, Dense, Activation, Convolution2D, MaxPooling2D  # Capas de la red
from keras.optimizers import Adam  # Optimizador
from keras import backend as K  # Herramientas adicionales para manejar sesiones

# Limpiar cualquier sesión previa de Keras
K.clear_session()

#--------------------------------- Configuración ---------------------------------
datos_entrenamiento = r'D:\SEMESTRE7\Dedos\Fotos\Entrenamiento'
datos_validacion = r'D:\SEMESTRE7\Dedos\Fotos\Validacion'

# Parámetros
iteraciones = 20  # Número de iteraciones para ajustar nuestro modelo
altura, longitud = 200, 200  # Tamaño de las imágenes
batch_size = 1  # Tamaño de lote
pasos = 300 // batch_size  # Número de pasos por iteración
pasos_validacion = 300 // batch_size  # Pasos en validación
filtrosconv1 = 32
filtrosconv2 = 64
filtrosconv3 = 128
tam_filtro1 = (4, 4)
tam_filtro2 = (3, 3)
tam_filtro3 = (2, 2)
tam_pool = (2, 2)
clases = 5  # Número de clases (en este caso, vocales)
lr = 0.0005  # Tasa de aprendizaje

#-------------------------- Preprocesamiento de Imágenes -------------------------
preprocesamiento_entre = ImageDataGenerator(
    rescale=1. / 255,  # Normalización de píxeles
    shear_range=0.3,  # Inclinación
    zoom_range=0.3,  # Zoom aleatorio
    horizontal_flip=True  # Volteo horizontal
)

preprocesamiento_vali = ImageDataGenerator(
    rescale=1. / 255  # Solo normalización
)

# Directorios de imágenes para entrenamiento y validación
imagen_entreno = preprocesamiento_entre.flow_from_directory(
    datos_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

imagen_validacion = preprocesamiento_vali.flow_from_directory(
    datos_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

#-------------------------- Crear la Red Neuronal (CNN) -------------------------
cnn = Sequential()
# Primera capa de convolución
cnn.add(Convolution2D(filtrosconv1, tam_filtro1, padding='same', input_shape=(altura, longitud, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

# Segunda capa de convolución
cnn.add(Convolution2D(filtrosconv2, tam_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

# Tercera capa de convolución
cnn.add(Convolution2D(filtrosconv3, tam_filtro3, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tam_pool))

# Aplanado de la imagen
cnn.add(Flatten())
cnn.add(Dense(640, activation='relu'))  # Capa densa con 640 neuronas
cnn.add(Dropout(0.5))  # Dropout para evitar sobreajuste
cnn.add(Dense(clases, activation='softmax'))  # Capa de salida

#------------------------- Compilación y Entrenamiento --------------------------
# Compilar el modelo
optimizar = Adam(learning_rate=lr)
cnn.compile(loss='categorical_crossentropy', optimizer=optimizar, metrics=['accuracy'])

# Entrenar la red
cnn.fit(imagen_entreno, steps_per_epoch=pasos, epochs=iteraciones, validation_data=imagen_validacion, validation_steps=pasos_validacion)

# Guardar el modelo y los pesos
cnn.save('ModeloVocales.keras')
cnn.save_weights('pesosVocales.weights.h5')