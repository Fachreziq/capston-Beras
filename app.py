"""
app.py
Aplikasi Flask sederhana untuk demo klasifikasi jenis beras.
Pastikan model/best_model.keras dan model/labels.json sudah ada
(hasil menjalankan src/train.py) sebelum menjalankan app ini.

Jalankan: python app.py
Lalu buka: http://127.0.0.1:5000
"""

import json
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet

MODEL_PATH = Path("model/best_model.keras")
LABELS_PATH = Path("model/labels.json")
SUMMARY_PATH = Path("model/summary.json")
IMG_SIZE = (224, 224)  # fallback default, akan ditimpa otomatis dari summary.json

# PENTING: MobileNetV2 dan EfficientNetB0 butuh normalisasi input yang
# berbeda. Fungsi yang dipakai harus SAMA PERSIS dengan yang dipakai saat
# training model tersebut (lihat src/train.py), kalau tidak, prediksi bisa
# jadi acak/salah walau modelnya sendiri sudah bagus.
PREPROCESS_FN = {
    "MobileNetV2": preprocess_mobilenet,
    "EfficientNetB0": preprocess_efficientnet,
}
preprocess_fn = preprocess_mobilenet  # default, ditimpa otomatis di bawah

app = Flask(__name__)

model = None
class_names = []


def load_model_and_labels():
    global model, class_names, IMG_SIZE, preprocess_fn
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "Model belum ditemukan. Jalankan 'python src/train.py' terlebih dahulu."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f:
        class_names = json.load(f)

    # Ambil ukuran input & arsitektur model terbaik dari summary.json, supaya
    # ukuran gambar dan preprocessing selalu sinkron dengan hasil training.
    if SUMMARY_PATH.exists():
        with open(SUMMARY_PATH) as f:
            summary = json.load(f)
        if "img_size" in summary:
            IMG_SIZE = tuple(summary["img_size"])
            print(f"Ukuran input gambar terdeteksi dari summary.json: {IMG_SIZE}")
        best_model_name = summary.get("best_model")
        if best_model_name in PREPROCESS_FN:
            preprocess_fn = PREPROCESS_FN[best_model_name]
            print(f"Preprocessing terdeteksi untuk model terbaik: {best_model_name}")


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(image).astype("float32")
    arr = preprocess_fn(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    all_probs = None
    error = None

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            error = "Silakan pilih file gambar terlebih dahulu."
        else:
            try:
                image = Image.open(file.stream)
                arr = preprocess_image(image)
                probs = model.predict(arr)[0]
                idx = int(np.argmax(probs))

                prediction = class_names[idx]
                confidence = float(probs[idx]) * 100
                all_probs = sorted(
                    zip(class_names, [float(p) * 100 for p in probs]),
                    key=lambda x: x[1],
                    reverse=True,
                )
            except Exception as e:
                error = f"Gagal memproses gambar: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        all_probs=all_probs,
        error=error,
    )



# Load model saat aplikasi dijalankan oleh Gunicorn/Railway
load_model_and_labels()

if __name__ == "__main__":
    app.run(debug=False)

