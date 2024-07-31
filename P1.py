import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

# Generar imágenes de ejemplo
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
model = Sequential([
    Flatten(input_shape=(100, 100, 3)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, verbose=1)

# Función para cargar y predecir imagen
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
        prediction = model.predict(img_array)
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

root.mainloop()
