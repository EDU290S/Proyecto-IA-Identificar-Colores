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
from PIL import Image, ImageTk, ImageDraw
import os

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

# Diccionario para los colores BGR usados en OpenCV
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

# Función para obtener los límites HSV para un color dado
def get_limits(color_name):
    return color_ranges[color_name]

# Función para crear una imagen de 100x100 píxeles de un color específico
def create_image_with_color(color):
    img = Image.new('RGB', (100, 100), color=color)
    return np.array(img)



    """
    Seccion de implementacion de epocas
    
    """
# Función para entrenar el modelo de detección de colores
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
        for _ in range(50):  # Crear 50 imágenes para cada color
            X_train.append(create_image_with_color(color))
            y_train.append(idx)
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Definición de la estructura del modelo neuronal
    color_model = Sequential([
        Flatten(input_shape=(100, 100, 3)),  # Capa de entrada: 30000 neuronas (100x100x3)
        Dense(128, activation='relu'),       # Capa oculta: 128 neuronas
        Dense(len(colors), activation='softmax')  # Capa de salida: 8 neuronas (una por cada color)
    ])
    
    color_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Callback personalizado para registrar la información de cada época
    class CustomCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            global training_details
            details = f'Epoch {epoch+1}: loss = {logs["loss"]:.4f}, accuracy = {logs["accuracy"]:.4f}'
            training_details.append(details)
            print(details)
            # Imprime los pesos de la primera capa
            print("Pesos de la primera capa:", self.model.layers[1].get_weights()[0])
            app.update_training_info()

    # Entrenamiento del modelo con los datos generados
    color_model.fit(X_train, y_train, epochs=epochs, callbacks=[CustomCallback()], verbose=1)
    color_model.save('color_model.h5')  # Guardar el modelo entrenado
    return color_model

# Función para cargar un modelo entrenado desde el disco
def load_trained_model(filepath='color_model.h5'):
    if os.path.exists(filepath):
        return load_model(filepath)
    return None

# Función para obtener los colores dominantes en una imagen usando KMeans
def get_dominant_colors(image, k=5):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    return dominant_colors

# Cargar el modelo MobileNetV2 preentrenado para la clasificación de imágenes
mobilenet_model = MobileNetV2(weights='imagenet')

# Función para clasificar una imagen usando MobileNetV2
def classify_image(image):
    image_resized = cv2.resize(image, (224, 224))
    image_array = img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    predictions = mobilenet_model.predict(image_array)
    results = decode_predictions(predictions, top=3)[0]
    return results

# Función para dibujar los colores dominantes en una barra horizontal
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

# Función para actualizar el marco de la cámara en tiempo real
def update_frame():
    ret, frame = app.cap.read()
    if ret:
        process_frame(frame)
    
    if app.camera_on:
        app.root.after(10, update_frame)

# Función para procesar un marco de la cámara y detectar colores
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

    shapes = detect_shapes_in_image(frame_copy)
    app.shape_count_label.config(text="\n".join([f"{shape_name}: {count}" for shape_name, count in shapes.items()]))
    
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

# Función para detectar colores en una imagen
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

# Función para detectar formas en una imagen
def detect_shapes_in_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = {"Circle": 0, "Square": 0, "Triangle": 0, "Rectangle": 0, "Polygon": 0, "Rectangle (Side)": 0}
    
    for contour in contours:
        shape = classify_shape(contour)
        shapes[shape] += 1
        
    return shapes

# Función para clasificar una forma en función de sus contornos
def classify_shape(contour):
    shape = "Unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)  # Aumenta la precisión del contorno
    
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        if ar >= 0.90 and ar <= 1.10:  # Relación de aspecto ajustada para mejor precisión
            shape = "Square"
        else:
            angle = cv2.minAreaRect(approx)[-1]
            shape = "Rectangle (Side)" if angle not in [-90, 0] else "Rectangle"
    elif len(approx) > 4:
        shape = "Circle"
    else:
        shape = "Polygon"
    
    return shape

# Función para cargar una imagen y predecir los colores en ella
def load_and_predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        if img is None:
            print(f"No se pudo cargar la imagen desde {file_path}")
            return
        img = resize_image(img, 800)  # Redimensionar la imagen a un ancho máximo de 800 píxeles
        # Entrenar el modelo con 100 épocas
        train_color_model(epochs=100)
        process_image(img)

# Función para redimensionar una imagen
def resize_image(img, max_width):
    height, width = img.shape[:2]
    if width > max_width:
        new_width = max_width
        new_height = int(height * (max_width / width))
        img = cv2.resize(img, (new_width, new_height))
    return img

# Función para procesar una imagen cargada y detectar colores
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

    shapes = detect_shapes_in_image(img_copy)
    app.shape_count_label.config(text="\n".join([f"{shape_name}: {count}" for shape_name, count in shapes.items()]))
    
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

def draw_rectangle(color, size=20):
    img = Image.new('RGB', (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, size-1, size-1), outline=color, width=3)
    draw.rectangle((3, 3, size-4, size-4), fill=color)
    return ImageTk.PhotoImage(img)

def draw_color_rectangles(colors, size=50):
    img = Image.new('RGB', (len(colors) * size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, color in enumerate(colors):
        draw.rectangle([i * size, 0, (i + 1) * size, size], fill=color)
    return ImageTk.PhotoImage(img)

# Clase principal de la aplicación
class ColorDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Colores y Formas")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f8f9fa')  # Fondo claro
        
        self.colors = ["Rojo", "Verde", "Azul", "Cian", "Magenta", "Amarillo", "Negro", "Blanco"]
        self.color_names = ["Rojo", "Verde", "Azul", "Cian", "Magenta", "Amarillo", "Negro", "Blanco"]

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f8f9fa')
        self.style.configure('TLabel', background='#f8f9fa', font=('Arial', 12))
        self.style.configure('TButton', font=('Arial', 12), background='#007bff', foreground='white')  # Fondo azul
        self.style.configure('Menu.TFrame', background='#6f42c1')  # Fondo morado para el menú
        self.style.configure('Shadow.TFrame', relief='groove', borderwidth=2)
        self.style.configure('Shadow.TLabel', background='#f8f9fa', font=('Arial', 12), relief='groove', borderwidth=2)
        
        self.menu_frame = ttk.Frame(self.root, width=200, style='Menu.TFrame')
        self.menu_frame.pack(fill=tk.Y, side=tk.LEFT, padx=10, pady=10)


        self.title_label = ttk.Label(self.menu_frame, text="C", font=('impact', 30), background='#6f42c1', foreground='white', anchor='center')
        self.title_label.pack(pady=10)
        self.title_label = ttk.Label(self.menu_frame, text="O", font=('impact', 30), background='#6f42c1', foreground='white', anchor='center')
        self.title_label.pack(pady=10)
        self.title_label = ttk.Label(self.menu_frame, text="L", font=('impact', 30), background='#6f42c1', foreground='white', anchor='center')
        self.title_label.pack(pady=10)
        self.title_label = ttk.Label(self.menu_frame, text="O", font=('impact', 30), background='#6f42c1', foreground='white', anchor='center')
        self.title_label.pack(pady=10)
        self.title_label = ttk.Label(self.menu_frame, text="R", font=('impact', 30), background='#6f42c1', foreground='white', anchor='center')
        self.title_label.pack(pady=10)
        self.title_label = ttk.Label(self.menu_frame, text="N", font=('impact', 25), background='#6f42c1', foreground='white', anchor='center')
        self.title_label.pack(pady=10)
        self.title_label = ttk.Label(self.menu_frame, text="E", font=('impact', 25), background='#6f42c1', foreground='white', anchor='center')
        self.title_label.pack(pady=10)
        self.title_label = ttk.Label(self.menu_frame, text="T", font=('impact', 25), background='#6f42c1', foreground='white', anchor='center')
        self.title_label.pack(pady=10)
        
        self.left_frame = ttk.Frame(self.root, style='Shadow.TFrame')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.right_frame = ttk.Frame(self.root, style='Shadow.TFrame')
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = ttk.Label(self.left_frame, anchor='center', style='Shadow.TLabel')
        self.video_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.color_bar_label = ttk.Label(self.left_frame, anchor='center', style='Shadow.TLabel')
        self.color_bar_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.color_labels_frame = ttk.Frame(self.left_frame)
        self.color_labels_frame.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.color_labels = []
        for color_name, color in zip(self.color_names, ["#ff0000", "#00ff00", "#0000ff", "#00ffff", "#ff00ff", "#ffff00", "#000000", "#ffffff"]):
            frame = ttk.Frame(self.color_labels_frame)
            label = ttk.Label(frame, text=color_name, width=10, font=('Arial', 10), background='#f8f9fa', anchor='center')
            label.pack(side=tk.TOP, padx=2, pady=2)
            rectangle = ttk.Label(frame, image=draw_rectangle(color), background='#f8f9fa', anchor='center')
            rectangle.pack(side=tk.BOTTOM, padx=2, pady=2)
            frame.pack(side=tk.LEFT, padx=5, pady=5)
            self.color_labels.append((label, rectangle))

        self.color_display_label = ttk.Label(self.left_frame, anchor='center')
        self.color_display_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        colors_bgr = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (0, 0, 0), (255, 255, 255)]
        color_rectangles_img = draw_color_rectangles(colors_bgr)
        self.color_display_label.imgtk = color_rectangles_img
        self.color_display_label.configure(image=color_rectangles_img)

        self.pixel_count_label = ttk.Label(self.right_frame, text="Pixeles detectados:", justify=tk.CENTER, anchor='center', style='Shadow.TLabel')
        self.pixel_count_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.shape_count_label = ttk.Label(self.right_frame, text="Formas detectadas:", justify=tk.CENTER, anchor='center', style='Shadow.TLabel')
        self.shape_count_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.classification_label = ttk.Label(self.right_frame, text="", justify=tk.CENTER, anchor='center', style='Shadow.TLabel')
        self.classification_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.training_info_label = ttk.Label(self.right_frame, text="Información de Entrenamiento:", justify=tk.CENTER, anchor='center', style='Shadow.TLabel')
        self.training_info_label.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.root)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.button_frame = ttk.Frame(self.menu_frame)
        self.button_frame.pack(padx=5, pady=10, fill=tk.X, side=tk.BOTTOM)

        self.toggle_camera_button = ttk.Button(self.button_frame, text="Activar/Desactivar Cámara", command=self.toggle_camera)
        self.toggle_camera_button.pack(padx=5, pady=5, fill=tk.X)

        self.load_button = ttk.Button(self.button_frame, text="Cargar Imagen", command=load_and_predict_image)
        self.load_button.pack(padx=5, pady=5, fill=tk.X)

        self.exit_button = ttk.Button(self.button_frame, text="Salir", command=root.quit)
        self.exit_button.pack(padx=5, pady=5, fill=tk.X)

        self.canvas_frame = ttk.Frame(self.right_frame)
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

# Función principal que ejecuta la aplicación
def main():
    global app
    root = tk.Tk()
    app = ColorDetectorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
