# Klasifikasi Jenis Beras Menggunakan Transfer Learning (MobileNetV2 vs EfficientNetB0)

Capstone Project — Mata Kuliah Kecerdasan Buatan (Computer Vision)

## 1. Deskripsi Proyek

Proyek ini bertujuan mengklasifikasikan **kualitas beras** (utuh/patah/campuran)
ke dalam beberapa kategori menggunakan pendekatan *transfer learning*. Dua
arsitektur dibandingkan untuk melihat mana yang memberikan performa terbaik
untuk task ini:

- **MobileNetV2** — ringan, cocok untuk deployment web/mobile.
- **EfficientNetB0** — lebih dalam, potensi akurasi lebih tinggi dengan trade-off
  ukuran model dan waktu inferensi.

Model terbaik dari hasil perbandingan kemudian di-deploy sebagai aplikasi web
sederhana berbasis **Flask**, di mana pengguna dapat mengunggah foto beras dan
sistem akan memprediksi kategorinya beserta tingkat keyakinan (confidence).

## 2. Sumber Dataset

Dataset: **Rice Images Dataset** (Kaggle)
Link: https://www.kaggle.com/datasets/shubhamcodez/rice-images-dataset

Dataset ini berisi citra beras dengan kategori kualitas (broken/full/mixed —
sesuaikan nama persisnya dengan folder yang muncul setelah kamu extract, karena
penamaan bisa berbeda dari deskripsi umum di halaman Kaggle). Ukurannya jauh
lebih kecil dibanding dataset varietas beras (Arborio dkk.), jadi cocok untuk
proses training yang ringan dan cepat.

> Catatan penting: setelah download & extract, jalankan `ls data/raw/` (atau
> buka foldernya) untuk melihat nama-nama kelas yang sebenarnya, lalu pastikan
> strukturnya `data/raw/<nama_kelas>/*.jpg` seperti contoh di bawah. Script
> `split_dataset.py` akan otomatis mendeteksi kelas apa pun nama foldernya —
> tidak perlu mengedit kode untuk ini.
>
> Karena datasetnya sudah relatif kecil, `MAX_PER_CLASS` di `split_dataset.py`
> di-set ke 400 (bisa dinaikkan kalau ternyata jumlah data per kelas sedikit,
> supaya kamu tidak kehabisan data training).

## 3. Struktur Folder

```
capstone-beras/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # dataset asli, dipisah per folder kelas
│   │   ├── Full/           # (nama folder sesuaikan hasil extract dataset)
│   │   ├── Broken/
│   │   └── Mixed/
│   └── processed/          # (opsional) hasil split train/val/test
├── notebooks/
│   └── eksplorasi_data.ipynb
├── src/
│   ├── train.py            # training & perbandingan 2 model
│   └── split_dataset.py    # membagi data jadi train/val/test
├── model/                  # model hasil training (.keras) & label mapping
├── templates/
│   └── index.html          # halaman web Flask
├── static/
│   └── style.css
└── app.py                  # aplikasi Flask untuk demo
```

## 4. Cara Menjalankan

### a. Setup environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### b. Siapkan dataset
Unduh dataset dari Kaggle, ekstrak ke `data/raw/` mengikuti struktur folder
per kelas di atas. Lalu jalankan:
```bash
python src/split_dataset.py
```
Script ini akan membagi data menjadi `data/processed/train`, `val`, dan `test`
(default 70/15/15).

### c. Training & perbandingan model
```bash
python src/train.py
```
Script ini akan:
1. Melatih model MobileNetV2 (transfer learning, base frozen lalu fine-tuning).
2. Melatih model EfficientNetB0 dengan skema yang sama.
3. Mengevaluasi keduanya di test set (accuracy, precision, recall, F1, confusion matrix).
4. Menyimpan kedua model beserta grafik perbandingan ke folder `model/`.
5. Menyimpan `model/best_model.keras` (model dengan akurasi test tertinggi) dan
   `model/labels.json` untuk dipakai aplikasi Flask.

### d. Menjalankan aplikasi web
```bash
python app.py
```
Buka `http://127.0.0.1:5000` di browser, unggah foto beras, dan lihat hasil
prediksinya.

## 5. Metodologi Singkat (untuk laporan)

- **Preprocessing**: resize ke 224x224, normalisasi pixel [0,1], augmentasi
  (rotasi, flip horizontal, zoom, brightness) hanya pada data training.
- **Arsitektur**: base model MobileNetV2 / EfficientNetB0 (pretrained ImageNet,
  `include_top=False`) + GlobalAveragePooling2D + Dense + Dropout + output
  softmax (jumlah unit mengikuti jumlah kelas kualitas beras pada dataset,
  otomatis terdeteksi dari struktur folder `data/raw/`).
- **Skema eksperimen**: 
  1. Feature extraction (base model frozen, hanya melatih head).
  2. Fine-tuning (unfreeze beberapa layer terakhir base model, learning rate
     kecil) — bandingkan hasil sebelum dan sesudah fine-tuning.
- **Metrik evaluasi**: accuracy, precision, recall, F1-score (macro avg), dan
  confusion matrix karena ini task klasifikasi multi-kelas seimbang.

## 6. Batasan yang Perlu Disebutkan di Laporan

- Model dilatih pada dataset dengan kondisi pencahayaan/latar relatif seragam;
  performa pada foto dengan latar belakang kompleks (misal foto dari HP di meja
  dapur) kemungkinan lebih rendah.
- Belum menangani kasus campuran beberapa jenis beras dalam satu foto.

## 7. Author
Nama: [isi nama kamu]
NIM: [isi NIM]
Kelas: [isi kelas]
