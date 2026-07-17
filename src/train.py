"""
train.py
Melatih dan membandingkan dua model transfer learning:
- MobileNetV2
- EfficientNetB0

untuk klasifikasi jenis beras. Menyimpan kedua model, grafik perbandingan,
laporan evaluasi (classification report + confusion matrix), dan model
terbaik (model/best_model.keras) beserta model/labels.json untuk dipakai
aplikasi Flask.

Jalankan: python src/train.py
"""

import gc
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_efficientnet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

FAST_MODE = True   # set False kalau mau kualitas maksimal (lebih lama)

if FAST_MODE:
    IMG_SIZE = (160, 160)     # lebih kecil dari 224x224 -> jauh lebih cepat
    BATCH_SIZE = 32           # dikecilkan dari 64 -> lebih aman untuk RAM terbatas
    EPOCHS_HEAD = 5
    EPOCHS_FINE_TUNE = 3      # set ke 0 untuk skip fine-tuning sama sekali
    LEARNING_RATE_HEAD = 1e-3
    LEARNING_RATE_FT = 1e-5
else:
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS_HEAD = 10
    EPOCHS_FINE_TUNE = 8
    LEARNING_RATE_HEAD = 1e-3
    LEARNING_RATE_FT = 1e-5


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
# PENTING: setiap arsitektur transfer learning punya skema normalisasi input
# yang berbeda (MobileNetV2 -> [-1, 1], EfficientNet -> punya normalisasi
# internal sendiri, JANGAN di-rescale manual lagi). Karena itu generator
# dibuat terpisah per arsitektur menggunakan preprocess_input resminya
# masing-masing, BUKAN rescale=1/255 yang seragam untuk semua model.
def build_generators(preprocess_fn):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_datagen.flow_from_directory(
        DATA_DIR / "train",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
    )
    val_gen = val_test_datagen.flow_from_directory(
        DATA_DIR / "val",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        DATA_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_gen, val_gen, test_gen


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(base_model_fn, num_classes):
    base_model = base_model_fn(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


# ---------------------------------------------------------------------------
# Training pipeline untuk satu arsitektur
# ---------------------------------------------------------------------------
def train_one_architecture(name, base_model_fn, train_gen, val_gen, num_classes):
    print(f"\n{'=' * 60}\nTraining: {name}\n{'=' * 60}")

    model, base_model = build_model(base_model_fn, num_classes)
    compile_model(model, LEARNING_RATE_HEAD)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2
        ),
    ]

    # Tahap 1: feature extraction
    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
    )

    # Tahap 2: fine-tuning — unfreeze sebagian layer terakhir base model
    # (dilewati kalau EPOCHS_FINE_TUNE = 0, untuk mode super cepat)
    if EPOCHS_FINE_TUNE > 0:
        base_model.trainable = True
        fine_tune_at = int(len(base_model.layers) * 0.7)
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        compile_model(model, LEARNING_RATE_FT)

        history_ft = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=EPOCHS_FINE_TUNE,
            callbacks=callbacks,
        )
        ft_acc = history_ft.history["accuracy"]
        ft_val_acc = history_ft.history["val_accuracy"]
        ft_loss = history_ft.history["loss"]
        ft_val_loss = history_ft.history["val_loss"]
    else:
        ft_acc = ft_val_acc = ft_loss = ft_val_loss = []

    # Gabungkan history untuk plotting
    combined_history = {
        "accuracy": history_head.history["accuracy"] + ft_acc,
        "val_accuracy": history_head.history["val_accuracy"] + ft_val_acc,
        "loss": history_head.history["loss"] + ft_loss,
        "val_loss": history_head.history["val_loss"] + ft_val_loss,
    }

    return model, combined_history


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------
def evaluate_model(model, test_gen, class_names, name):
    test_gen.reset()
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    print(f"\nClassification report — {name}:\n{report_text}")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / f"confusion_matrix_{name}.png")
    plt.close()

    with open(MODEL_DIR / f"classification_report_{name}.json", "w") as f:
        json.dump(report, f, indent=2)

    test_acc = report["accuracy"]
    return test_acc, report


def plot_comparison(histories: dict):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for name, hist in histories.items():
        plt.plot(hist["val_accuracy"], label=f"{name} (val)")
    plt.title("Perbandingan Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, hist in histories.items():
        plt.plot(hist["val_loss"], label=f"{name} (val)")
    plt.title("Perbandingan Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "comparison_mobilenet_vs_efficientnet.png")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    architectures = {
        "MobileNetV2": (MobileNetV2, preprocess_mobilenet),
        "EfficientNetB0": (EfficientNetB0, preprocess_efficientnet),
    }

    # Ambil daftar nama kelas sekali saja (tidak tergantung preprocessing)
    probe_gen = ImageDataGenerator().flow_from_directory(
        DATA_DIR / "train", target_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    class_names = list(probe_gen.class_indices.keys())
    num_classes = len(class_names)
    print(f"Kelas: {class_names}")

    results = {}
    histories = {}
    trained_models = {}

    for name, (base_fn, preprocess_fn) in architectures.items():
        # Generator dibuat ulang tiap arsitektur, dengan preprocessing_function
        # resmi masing-masing arsitektur (lihat catatan di build_generators)
        train_gen, val_gen, test_gen = build_generators(preprocess_fn)

        model, history = train_one_architecture(
            name, base_fn, train_gen, val_gen, num_classes
        )
        test_acc, report = evaluate_model(model, test_gen, class_names, name)

        model_path = MODEL_DIR / f"{name}.keras"
        model.save(model_path)

        results[name] = test_acc
        histories[name] = history
        trained_models[name] = model_path

        print(f"{name} — Test Accuracy: {test_acc:.4f} (saved to {model_path})")

        # Bersihkan memori sebelum lanjut ke arsitektur berikutnya. Tanpa ini,
        # RAM dari model & generator sebelumnya menumpuk dan bisa
        # menyebabkan ResourceExhaustedError/MemoryError di laptop dengan
        # RAM terbatas, terutama saat masuk tahap fine-tuning.
        del model, train_gen, val_gen, test_gen
        tf.keras.backend.clear_session()
        gc.collect()

    plot_comparison(histories)

    # Tentukan model terbaik
    best_name = max(results, key=results.get)
    print(f"\nModel terbaik: {best_name} (Test Accuracy: {results[best_name]:.4f})")

    best_model = tf.keras.models.load_model(trained_models[best_name])
    best_model.save(MODEL_DIR / "best_model.keras")

    with open(MODEL_DIR / "labels.json", "w") as f:
        json.dump(class_names, f, indent=2)

    with open(MODEL_DIR / "summary.json", "w") as f:
        json.dump(
            {
                "results": results,
                "best_model": best_name,
                "class_names": class_names,
                "img_size": list(IMG_SIZE),
            },
            f,
            indent=2,
        )

    print("\nSelesai. Model terbaik disimpan di model/best_model.keras")
    print("Ringkasan hasil ada di model/summary.json")
    print("Gunakan grafik dan classification report di folder model/ untuk laporan.")


if __name__ == "__main__":
    main()