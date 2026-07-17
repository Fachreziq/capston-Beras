"""
split_dataset.py
Membagi dataset mentah di data/raw/<kelas>/*.jpg menjadi
data/processed/{train,val,test}/<kelas>/*.jpg

Jalankan: python src/split_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42
MAX_PER_CLASS = 200   # dikecilkan lagi supaya training cepat
                       # naikkan (400-1000+) kalau nanti mau kualitas lebih baik
                       # set None kalau mau pakai semua data


def find_class_root(raw_dir: Path) -> Path:
    """
    Mendeteksi lokasi folder kelas secara otomatis.

    Menangani kasus umum saat extract dataset Kaggle: kadang seluruh isi
    dataset masuk ke satu folder pembungkus (mis. data/raw/rice_images/...)
    alih-alih folder kelas langsung di bawah data/raw/. Fungsi ini akan
    menelusuri ke bawah selama hanya ada SATU folder non-tersembunyi di
    level tersebut.
    """
    current = raw_dir
    while True:
        subdirs = [
            d for d in current.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if len(subdirs) == 1:
            # Kemungkinan folder pembungkus tunggal -> masuk lebih dalam
            current = subdirs[0]
            continue
        break
    return current


def main():
    random.seed(RANDOM_SEED)

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Folder {RAW_DIR} tidak ditemukan. Pastikan dataset sudah "
            f"diekstrak ke data/raw/<nama_kelas>/*.jpg"
        )

    class_root = find_class_root(RAW_DIR)
    if class_root != RAW_DIR:
        print(f"Folder pembungkus terdeteksi, membaca kelas dari: {class_root}")

    classes = sorted([
        d.name for d in class_root.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    if not classes:
        raise ValueError(
            f"Tidak ada folder kelas ditemukan di {class_root}. "
            f"Pastikan strukturnya data/raw/<nama_kelas>/*.jpg"
        )

    print(f"Ditemukan {len(classes)} kelas: {classes}")

    for split in ["train", "val", "test"]:
        for cls in classes:
            (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    summary = {}
    for cls in classes:
        images = list((class_root / cls).glob("*"))
        images = [p for p in images if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        random.shuffle(images)

        if MAX_PER_CLASS:
            images = images[:MAX_PER_CLASS]

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ]:
            for img_path in split_imgs:
                dest = OUT_DIR / split_name / cls / img_path.name
                shutil.copy2(img_path, dest)

        summary[cls] = {
            "total": n,
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }

    print("\nRingkasan pembagian data:")
    for cls, counts in summary.items():
        print(f"  {cls}: {counts}")

    print(f"\nSelesai. Data hasil split ada di: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
