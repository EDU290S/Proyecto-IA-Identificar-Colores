import cv2
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import time

# Inicializar el modelo preentrenado
model = MobileNetV2(weights='imagenet')

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
    
    predictions = model.predict(image_array)
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

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

if __name__ == "__main__":
    main()
