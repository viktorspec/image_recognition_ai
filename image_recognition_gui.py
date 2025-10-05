import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tkinter import Tk, Label, Button, filedialog, Toplevel
from PIL import Image, ImageTk
import threading
import time

# Загружаем модель один раз
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def recognize_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0]
    return decoded[1], decoded[2] * 100

def analyze_with_loading(path):
    # Создаём всплывающее окно с индикатором загрузки
    loading_win = Toplevel(root)
    loading_win.title("Анализ изображения...")
    loading_win.geometry("250x100")
    loading_win.configure(bg="#f4f6f8")
    Label(loading_win, text="🤖 Распознаю...", bg="#f4f6f8", font=("Arial", 14, "bold")).pack(pady=20)

    # Анимация точек "..."
    running = True
    def animate():
        dots = 0
        while running:
            loading_label.config(text="🤖 Распознаю" + "." * dots)
            dots = (dots + 1) % 4
            time.sleep(0.5)

    loading_label = Label(loading_win, text="🤖 Распознаю", bg="#f4f6f8", font=("Arial", 14))
    loading_label.pack()

    anim_thread = threading.Thread(target=animate)
    anim_thread.start()

    # Распознавание в отдельном потоке
    def process():
        nonlocal running
        label, confidence = recognize_image(path)
        running = False
        loading_win.destroy()
        result_label.config(
            text=f"🧠 {label} ({confidence:.2f}%)",
            fg="green",
            font=("Arial", 14, "bold")
        )

    threading.Thread(target=process).start()

def open_image():
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        result_label.config(text="")

        # Запуск распознавания с анимацией
        analyze_with_loading(file_path)

# Интерфейс
root = Tk()
root.title("AI Image Recognition — by Vitya ❤️")
root.geometry("420x500")
root.configure(bg="#f4f6f8")

Label(root, text="🤖 Распознавание изображений", bg="#f4f6f8", font=("Arial", 16, "bold")).pack(pady=10)

image_label = Label(root, bg="#f4f6f8")
image_label.pack()

Button(
    root,
    text="Выбрать изображение",
    command=open_image,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12, "bold"),
    relief="flat",
    padx=10, pady=5
).pack(pady=15)

result_label = Label(root, text="", bg="#f4f6f8", font=("Arial", 14))
result_label.pack(pady=10)

Label(root, text="powered by TensorFlow & MobileNetV2", bg="#f4f6f8", fg="#888", font=("Arial", 9)).pack(side="bottom", pady=5)

root.mainloop()


