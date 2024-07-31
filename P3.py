import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import time

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    hue = hsvC[0][0][0]  # Get the hue value

    # Handle red hue wrap-around
    if hue >= 165:  # Upper limit for divided red hue
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:  # Lower limit for divided red hue
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)

    return lowerLimit, upperLimit

# Entrenamiento del modelo auxiliar
def create_image_with_color(color):
    img = Image.new('RGB', (100, 100), color=color)
    return np.array(img)

# Crear un conjunto de datos simple
X_train = []
y_train = []

# Imágenes con el color rojo
for _ in range(50):
    X_train.append(create_image_with_color((255, 0, 0)))
    y_train.append(1)  # Etiqueta 1 para imágenes con el color rojo

# Imágenes sin el color rojo
for _ in range(50):
    X_train.append(create_image_with_color((0, 255, 0)))
    y_train.append(0)  # Etiqueta 0 para imágenes sin el color rojo

X_train = np.array(X_train)
y_train = np.array(y_train)

# Construir el modelo
color_model = Sequential([
    Flatten(input_shape=(100, 100, 3)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

color_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = color_model.fit(X_train, y_train, epochs=10, verbose=1)

# Inicializar el modelo preentrenado de MobileNetV2
mobilenet_model = MobileNetV2(weights='imagenet')

def get_dominant_colors(image, k=5):
    """Identificar colores dominantes en la imagen."""
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

def classify_image(image):
    """Clasificar la imagen utilizando MobileNetV2."""
    image_resized = cv2.resize(image, (224, 224))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    
    predictions = mobilenet_model.predict(image_array)
    results = decode_predictions(predictions, top=3)[0]
    return results

def draw_dominant_colors(colors, ax):
    """Dibujar los colores dominantes en una ventana separada."""
    bar_height = 50
    bar_width = 300
    bar = np.zeros((bar_height, bar_width, 3), dtype="uint8")
    
    start_x = 0
    for color in colors:
        end_x = start_x + (bar_width // len(colors))
        bar[:, start_x:end_x] = color
        start_x = end_x

    ax.clear()
    ax.imshow(bar)
    ax.axis('off')
    ax.set_title('Colores Dominantes')

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return
    
    plt.ion()  # Modo interactivo
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el fotograma.")
            break
        
        start_time = time.time()
        
        # Clasifica los objetos en el fotograma
        classifications = classify_image(frame)
        label = classifications[0][1]  # Toma la etiqueta del primer resultado
        confidence = classifications[0][2]  # Toma la confianza del primer resultado
        
        # Identifica los colores dominantes
        colors = get_dominant_colors(frame)
        
        # Detectar el color amarillo en la imagen
        yellow = [0, 255, 255]  # yellow in BGR colorspace
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_limit, upper_limit = get_limits(color=yellow)
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Dibuja la clasificación en el fotograma
        cv2.putText(frame, f'{label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Convierte el fotograma de BGR a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Actualiza la imagen mostrada y los colores dominantes
        ax1.clear()
        ax1.imshow(frame_rgb)
        ax1.axis('off')
        ax1.set_title(f'{label}: {confidence:.2f}')
        
        draw_dominant_colors(colors, ax2)
        
        ax3.clear()
        ax3.imshow(mask, cmap='gray')
        ax3.axis('off')
        ax3.set_title('Detección de Color Amarillo')
        
        plt.draw()
        plt.pause(0.001)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        print(f'FPS: {fps:.2f}')

        # Presiona 'q' para salir
        if plt.waitforbuttonpress(0.001) and plt.get_fignums() == []:
            break

    cap.release()
    plt.ioff()
    plt.show()

def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((100, 100))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = color_model.predict(img_array)
        result = "Color Detectado" if prediction[0][0] > 0.5 else "Color No Detectado"
        result_label.config(text=result)

# Crear ventana principal
root = tk.Tk()
root.title("Detector de Color")

# Botón para cargar imagen
load_button = tk.Button(root, text="Cargar Imagen", command=load_and_predict_image)
load_button.pack()

# Etiqueta para mostrar resultado
result_label = tk.Label(root, text="")
result_label.pack()

# Etiqueta para mostrar información del entrenamiento
train_info = tk.Label(root, text=f"Entrenamiento completado en 10 épocas")
train_info.pack()

# Botón para iniciar detección en tiempo real
start_button = tk.Button(root, text="Iniciar Detección en Tiempo Real", command=main)
start_button.pack()

root.mainloop()
