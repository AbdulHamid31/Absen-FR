import streamlit as st
import cv2
import numpy as np
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import os
import zipfile
import gdown
from utils import encode_faces

st.set_page_config(page_title="Absensi Wajah", layout="centered")
st.title("📷 Absensi Pegawai - Face Recognition")

# Ganti ID Google Drive kamu di bawah ini
DATASET_ZIP_URL = "https://drive.google.com/uc?id=PASTE_ID_DI_SINI"  # Ganti ID dengan ID kamu
DATASET_ZIP_PATH = "dataset.zip"
DATASET_DIR = "dataset"

if st.button("🔄 Sinkronisasi Dataset dari Google Drive"):
    try:
        if os.path.exists(DATASET_DIR):
            st.info("Menghapus dataset lama...")
            os.system(f"rm -rf {DATASET_DIR}")

        st.info("📥 Mengunduh dataset...")
        gdown.download(DATASET_ZIP_URL, DATASET_ZIP_PATH, quiet=False)

        st.info("📂 Mengekstrak dataset...")
        with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATASET_DIR)

        st.info("🧠 Memproses wajah...")
        encode_faces(DATASET_DIR)
        st.success("✅ Sinkronisasi selesai!")
    except Exception as e:
        st.error(f"❌ Gagal sinkronisasi: {e}")

if os.path.exists("embeddings.pickle"):
    with open("embeddings.pickle", "rb") as f:
        data = pickle.load(f)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, faces)

        for face_encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], face_encoding)
            name = "Tidak Dikenal"

            if True in matches:
                matched_idx = matches.index(True)
                name = data["names"][matched_idx]

            st.image(frame, channels="BGR", caption=f"Deteksi: {name}")

            if name != "Tidak Dikenal":
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df = pd.DataFrame([[name, now]], columns=["Nama", "Waktu"])
                if os.path.exists("absensi.csv"):
                    df.to_csv("absensi.csv", mode='a', header=False, index=False)
                else:
                    df.to_csv("absensi.csv", index=False)
                st.success(f"✅ Absensi dicatat untuk {name} pada {now}")
    else:
        st.error("❌ Kamera tidak tersedia.")
else:
    st.warning("⚠️ Dataset belum disinkronisasi. Klik tombol di atas dahulu.")

if st.button("📄 Lihat Riwayat Absensi"):
    if os.path.exists("absensi.csv"):
        df = pd.read_csv("absensi.csv")
        st.dataframe(df)
    else:
        st.info("Belum ada data absensi.")
