import cv2
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import time
import matplotlib.pyplot as plt

def get_limits(color):
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]
    if hue >= 165:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([180, 255, 255], dtype=np.uint8)
    elif hue <= 15:
        lowerLimit = np.array([0, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    else:
        lowerLimit = np.array([hue - 10, 100, 100], dtype=np.uint8)
        upperLimit = np.array([hue + 10, 255, 255], dtype=np.uint8)
    return lowerLimit, upperLimit

def create_image_with_color(color):
    img = Image.new('RGB', (100, 100), color=color)
    return np.array(img)

X_train = []
y_train = []

for _ in range(50):
    X_train.append(create_image_with_color((255, 0, 0)))
    y_train.append(1)

for _ in range(50):
    X_train.append(create_image_with_color((0, 255, 0)))
    y_train.append(0)

X_train = np.array(X_train)
y_train = np.array(y_train)

color_model = Sequential([
    Flatten(input_shape=(100, 100, 3)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

color_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = color_model.fit(X_train, y_train, epochs=10, verbose=1)

mobilenet_model = MobileNetV2(weights='imagenet')

def get_dominant_colors(image, k=5):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

def classify_image(image):
    image_resized = cv2.resize(image, (224, 224))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    predictions = mobilenet_model.predict(image_array)
    results = decode_predictions(predictions, top=3)[0]
    return results

def draw_dominant_colors(colors, ax):
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

def update_frame():
    ret, frame = app.cap.read()
    if ret:
        masks = detect_colors_in_image(frame, app.colors)
        for color, mask, color_name in zip(app.colors, masks, app.color_names):
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        classifications = classify_image(frame)
        label = classifications[0][1]
        confidence = classifications[0][2]
        cv2.putText(frame, f'{label}: {confidence:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        colors = get_dominant_colors(frame)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        app.video_label.imgtk = imgtk
        app.video_label.configure(image=imgtk)
        
        app.ax1.clear()
        app.ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        app.ax1.axis('off')
        app.ax1.set_title(f'{label}: {confidence:.2f}')
        
        draw_dominant_colors(colors, app.ax2)
        
        app.ax3.clear()
        app.ax3.imshow(mask, cmap='gray')
        app.ax3.axis('off')
        app.ax3.set_title('Detección de Color Amarillo')
        
        plt.draw()
        plt.pause(0.001)
    
    if app.camera_on:
        app.root.after(10, update_frame)

def detect_colors_in_image(image, colors):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = []
    for color in colors:
        lower_limit, upper_limit = get_limits(color)
        mask = cv2.inRange(hsv_image, lower_limit, upper_limit)
        masks.append(mask)
    return masks

def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((100, 100))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prediction = color_model.predict(img_array)
        result = "Color Detectado" if prediction[0][0] > 0.5 else "Color No Detectado"
        app.result_label.config(text=result)

def create_model():
    model = Sequential([
        Flatten(input_shape=(100, 100, 3)),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_color_model(model, colors):
    X_train = np.array(colors)
    y_train = np.arange(len(colors))
    X_train = X_train / 255.0
    y_train = to_categorical(y_train, num_classes=len(colors))
    model.fit(X_train, y_train, epochs=100, verbose=1)

def save_model(model, filepath='color_model.h5'):
    model.save(filepath)

def load_trained_model(filepath='color_model.h5'):
    if os.path.exists(filepath):
        return load_model(filepath)
    return None

class ColorDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Colores")
        
        self.colors = [
            [255, 0, 0],   # Rojo
            [0, 255, 0],   # Verde
            [0, 0, 255],   # Azul
            [0, 255, 255], # Amarillo
            [255, 0, 255], # Magenta
            [255, 255, 0], # Cian
            [0, 0, 0],     # Negro
            [255, 255, 255], # Blanco
        ]

        self.color_names = ["Rojo", "Verde", "Azul", "Amarillo", "Magenta", "Cian", "Negro", "Blanco"]
        
        self.model = load_trained_model()
        if self.model is None:
            self.model = create_model()
            train_color_model(self.model, self.colors)
            save_model(self.model)
        
        self.camera_on = False
        
        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.color_labels = [tk.Label(root, text=color_name, bg='#%02x%02x%02x' % tuple(color), width=20) for color, color_name in zip(self.colors, self.color_names)]
        for label in self.color_labels:
            label.pack(side=tk.LEFT)
        
        self.toggle_camera_button = tk.Button(root, text="Activar/Desactivar Cámara", command=self.toggle_camera)
        self.toggle_camera_button.pack(pady=10)

        self.load_button = tk.Button(root, text="Cargar Imagen", command=load_and_predict_image)
        self.load_button.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

        self.train_info = tk.Label(root, text=f"Entrenamiento completado en 10 épocas")
        self.train_info.pack()
        
        self.cap = None

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(18, 6))
        plt.ion()
    
    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.cap.release()
            self.video_label.config(image='')
        else:
            self.cap = cv2.VideoCapture(0)
            self.camera_on = True
            update_frame()

def main():
    global app
    root = tk.Tk()
    app = ColorDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
