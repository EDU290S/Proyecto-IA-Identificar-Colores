import cv2
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Definimos los límites HSV para los colores específicos
color_ranges = {
    "Rojo": ([0, 120, 70], [10, 255, 255]),
    "Rojo2": ([170, 120, 70], [180, 255, 255]),  # Rojo tiene dos rangos en HSV
    "Verde": ([36, 100, 100], [86, 255, 255]),
    "Azul": ([94, 80, 2], [126, 255, 255]),
    "Cian": ([78, 100, 100], [94, 255, 255]),  # Ajustado para separar mejor cian y azul
    "Magenta": ([140, 100, 100], [170, 255, 255]),
    "Amarillo": ([25, 100, 100], [35, 255, 255]),
    "Negro": ([0, 0, 0], [180, 255, 30]),
    "Blanco": ([0, 0, 200], [180, 20, 255])
}

color_bgr = {
    "Rojo": (0, 0, 255),
    "Verde": (0, 255, 0),
    "Azul": (255, 0, 0),
    "Cian": (255, 255, 0),
    "Magenta": (255, 0, 255),
    "Amarillo": (0, 255, 255),
    "Negro": (0, 0, 0),
    "Blanco": (255, 255, 255)
}

# Variable global para almacenar los detalles de entrenamiento
training_details = []

def get_limits(color_name):
    return color_ranges[color_name]

def create_image_with_color(color):
    img = Image.new('RGB', (100, 100), color=color)
    return np.array(img)

def train_color_model(epochs=10):
    X_train = []
    y_train = []
    colors = [
        (255, 0, 0),     # Rojo
        (0, 255, 0),     # Verde
        (0, 0, 255),     # Azul
        (0, 255, 255),   # Cian
        (255, 0, 255),   # Magenta
        (255, 255, 0),   # Amarillo
        (0, 0, 0),       # Negro
        (255, 255, 255)  # Blanco
    ]

    for idx, color in enumerate(colors):
        for _ in range(50):
            X_train.append(create_image_with_color(color))
            y_train.append(idx)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    color_model = Sequential([
        Flatten(input_shape=(100, 100, 3)),
        Dense(128, activation='relu'),
        Dense(len(colors), activation='softmax')
    ])
    
    color_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            global training_details
            details = f'Epoch {epoch+1}: loss = {logs["loss"]:.4f}, accuracy = {logs["accuracy"]:.4f}'
            training_details.append(details)
            print(details)
            # Imprime los pesos de la primera capa
            print("Pesos de la primera capa:", self.model.layers[1].get_weights()[0])
            app.update_training_info()

    color_model.fit(X_train, y_train, epochs=epochs, callbacks=[CustomCallback()], verbose=1)
    color_model.save('color_model.h5')
    return color_model

def load_trained_model(filepath='color_model.h5'):
    if os.path.exists(filepath):
        return load_model(filepath)
    return None

def get_dominant_colors(image, k=5):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

mobilenet_model = MobileNetV2(weights='imagenet')

def classify_image(image):
    image_resized = cv2.resize(image, (224, 224))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    predictions = mobilenet_model.predict(image_array)
    results = decode_predictions(predictions, top=3)[0]
    return results

def draw_dominant_colors(colors):
    bar_height = 50
    bar_width = 300
    bar = np.zeros((bar_height, bar_width, 3), dtype="uint8")
    start_x = 0
    for color in colors:
        end_x = start_x + (bar_width // len(colors))
        bar[:, start_x:end_x] = color
        start_x = end_x
    return bar

def update_frame():
    ret, frame = app.cap.read()
    if ret:
        process_frame(frame)
    
    if app.camera_on:
        app.root.after(10, update_frame)

def process_frame(frame):
    masks = detect_colors_in_image(frame, app.colors)
    frame_copy = frame.copy()
    color_pixel_counts = {color_name: 0 for color_name in app.color_names}
    grouped_contours = {color_name: [] for color_name in app.color_names}
    
    for color_name, mask in zip(app.color_names, masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 500:  # Filtro para eliminar pequeñas detecciones
                color_pixel_counts[color_name] += cv2.countNonZero(mask[y:y+h, x:x+w])
                grouped_contours[color_name].append((x, y, w, h))
    
    for color_name, rects in grouped_contours.items():
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(frame_copy, (x, y), (x+w, y+h), color_bgr[color_name], 2)
            text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            text_color = (0, 0, 0) if color_name != "Negro" else (255, 255, 255)
            cv2.putText(frame_copy, color_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    app.pixel_count_label.config(text="\n".join([f"{color_name}: {count}" for color_name, count in color_pixel_counts.items()]))

    classifications = classify_image(frame)
    label = classifications[0][1]
    confidence = classifications[0][2]
    
    colors = get_dominant_colors(frame)
    color_bar = draw_dominant_colors(colors)
    
    color_bar_img = Image.fromarray(color_bar)
    color_bar_imgtk = ImageTk.PhotoImage(image=color_bar_img)
    app.color_bar_label.imgtk = color_bar_imgtk
    app.color_bar_label.configure(image=color_bar_imgtk)

    img = Image.fromarray(cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    app.video_label.imgtk = imgtk
    app.video_label.configure(image=imgtk)
    
    app.classification_label.config(text=f'{label}: {confidence:.2f}')

def detect_colors_in_image(image, colors):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = []
    for color_name in colors:
        lower_limit, upper_limit = get_limits(color_name)
        mask = cv2.inRange(hsv_image, np.array(lower_limit), np.array(upper_limit))
        if color_name == "Rojo":
            lower_limit2, upper_limit2 = get_limits("Rojo2")
            mask2 = cv2.inRange(hsv_image, np.array(lower_limit2), np.array(upper_limit2))
            mask = mask + mask2
        masks.append(mask)
    return masks

def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            print(f"No se pudo cargar la imagen desde {file_path}")
            return
        img = resize_image(img, 800)  # Redimensionar la imagen a un ancho máximo de 800 píxeles
        # Entrenar el modelo con 100 épocas
        train_color_model(epochs=10)
        process_image(img)

def resize_image(img, max_width):
    height, width = img.shape[:2]
    if width > max_width:
        new_width = max_width
        new_height = int(height * (max_width / width))
        img = cv2.resize(img, (new_width, new_height))
    return img

def process_image(img):
    masks = detect_colors_in_image(img, app.colors)
    img_copy = img.copy()
    color_pixel_counts = {color_name: 0 for color_name in app.color_names}
    grouped_contours = {color_name: [] for color_name in app.color_names}
    
    for color_name, mask in zip(app.color_names, masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 500:  # Filtro para eliminar pequeñas detecciones
                color_pixel_counts[color_name] += cv2.countNonZero(mask[y:y+h, x:x+w])
                grouped_contours[color_name].append((x, y, w, h))
    
    for color_name, rects in grouped_contours.items():
        for rect in rects:
            x, y, w, h = rect
            cv2.rectangle(img_copy, (x, y), (x+w, y+h), color_bgr[color_name], 2)
            text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x + (w - text_size[0]) // 2
            text_y = y + (h + text_size[1]) // 2
            text_color = (0, 0, 0) if color_name != "Negro" else (255, 255, 255)
            cv2.putText(img_copy, color_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    app.pixel_count_label.config(text="\n".join([f"{color_name}: {count}" for color_name, count in color_pixel_counts.items()]))
    
    classifications = classify_image(img)
    label = classifications[0][1]
    confidence = classifications[0][2]
    app.classification_label.config(text=f'{label}: {confidence:.2f}')
    
    colors = get_dominant_colors(img)
    color_bar = draw_dominant_colors(colors)
    
    color_bar_img = Image.fromarray(color_bar)
    color_bar_imgtk = ImageTk.PhotoImage(image=color_bar_img)
    app.color_bar_label.imgtk = color_bar_imgtk
    app.color_bar_label.configure(image=color_bar_imgtk)

    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    imgtk = ImageTk.PhotoImage(image=img_pil)
    app.video_label.imgtk = imgtk
    app.video_label.configure(image=imgtk)

class ColorDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Colores")
        self.root.geometry("1000x800")
        self.root.configure(bg='#2e2e2e')  # Fondo oscuro
        
        self.colors = ["Rojo", "Verde", "Azul", "Cian", "Magenta", "Amarillo", "Negro", "Blanco"]
        self.color_names = ["Rojo", "Verde", "Azul", "Cian", "Magenta", "Amarillo", "Negro", "Blanco"]

        self.paned_window = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, bg='#2e2e2e')
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        self.left_frame = tk.Frame(self.paned_window, bg='#2e2e2e')
        self.paned_window.add(self.left_frame, minsize=600)
        
        self.right_frame = tk.Frame(self.paned_window, bg='#2e2e2e')
        self.paned_window.add(self.right_frame, minsize=400)

        self.video_label = tk.Label(self.left_frame, bg='#2e2e2e')
        self.video_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.color_bar_label = tk.Label(self.left_frame, bg='#2e2e2e')
        self.color_bar_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.color_labels_frame = tk.Frame(self.left_frame, bg='#2e2e2e')
        self.color_labels_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.color_labels = [tk.Label(self.color_labels_frame, text=color_name, bg='#%02x%02x%02x' % tuple(color), width=20, fg='white', font=('Arial', 10)) for color, color_name in zip([(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0), (255, 255, 255)], self.color_names)]
        for label in self.color_labels:
            label.pack(side=tk.LEFT, padx=2)

        self.pixel_count_label = tk.Label(self.right_frame, text="Pixeles detectados:", bg='#2e2e2e', fg='white', justify=tk.LEFT)
        self.pixel_count_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.classification_label = tk.Label(self.right_frame, text="", bg='#2e2e2e', fg='white', justify=tk.LEFT)
        self.classification_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.training_info_label = tk.Label(self.right_frame, text="Información de Entrenamiento:", bg='#2e2e2e', fg='white', justify=tk.LEFT)
        self.training_info_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.button_frame = tk.Frame(self.right_frame, bg='#2e2e2e')
        self.button_frame.pack(padx=5, pady=10, fill=tk.X, side=tk.BOTTOM)

        self.toggle_camera_button = tk.Button(self.button_frame, text="Activar/Desactivar Cámara", command=self.toggle_camera, bg='#007bff', fg='white')
        self.toggle_camera_button.pack(padx=5, pady=5, fill=tk.X)

        self.load_button = tk.Button(self.button_frame, text="Cargar Imagen", command=load_and_predict_image, bg='#007bff', fg='white')
        self.load_button.pack(padx=5, pady=5, fill=tk.X)

        self.exit_button = tk.Button(self.button_frame, text="Salir", command=root.quit, bg='#dc3545', fg='white')
        self.exit_button.pack(padx=5, pady=5, fill=tk.X)

        self.canvas_frame = tk.Frame(self.right_frame, bg='#2e2e2e')
        self.canvas_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.color_model = load_trained_model()
        if self.color_model is None:
            self.color_model = train_color_model()
        
        self.camera_on = False
        self.cap = None

    def toggle_camera(self):
        if self.camera_on:
            self.camera_on = False
            self.cap.release()
            self.video_label.config(image='')
        else:
            self.cap = cv2.VideoCapture(0)
            self.camera_on = True
            update_frame()

    def update_training_info(self):
        global training_details
        info_text = "Información de Entrenamiento:\n" + "\n".join(training_details[-10:])
        self.training_info_label.config(text=info_text)

def main():
    global app
    root = tk.Tk()
    app = ColorDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
