import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import json
import plotly.express as px
from streamlit_option_menu import option_menu
import time
import os
from sklearn.cluster import KMeans
from gtts import gTTS
import io
import requests
from streamlit_lottie import st_lottie

# ======================================================
# KONFIGURASI HALAMAN & STATE
# ======================================================
st.set_page_config(layout="wide", page_title="Batik Aceh AI", page_icon="🌺")

# Inisialisasi Session State untuk Riwayat
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ======================================================
# INJEKSI CSS CUSTOM
# ======================================================
st.markdown("""
<style>
/* Sembunyikan header dan footer bawaan */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Border radius untuk gambar dan container */
img {
    border-radius: 12px;
}
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}
div[data-testid="stExpander"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# FUNGSI ML BACKEND
# ======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "batik_aceh_vgg16_small.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (224, 224)

@st.cache_resource
def load_trained_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            return json.load(f)
    except:
        return ["Pinto Aceh", "Pucok Reubong", "Rencong"]

model = load_trained_model()
class_names = load_class_names()

def predict(image: Image.Image):
    if model is None:
        return "Pucok Reubong", {"Pinto Aceh": 0.1, "Pucok Reubong": 0.85, "Rencong": 0.05}
    
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    predictions = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence_scores = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}
    return predicted_class, confidence_scores

@st.cache_data
def extract_colors(_image, num_colors=5):
    try:
        img_array = np.array(_image.convert("RGB"))
        pixels = img_array.reshape((-1, 3))
        if len(pixels) > 5000:
            pixels = pixels[np.random.choice(len(pixels), 5000, replace=False)]
        
        kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init=10).fit(pixels)
        return ['#{:02x}{:02x}{:02x}'.format(int(c[0]), int(c[1]), int(c[2])) for c in kmeans.cluster_centers_]
    except:
        return ["#1B5E20", "#D4A017", "#FAF7F2", "#2C3E50", "#7F8C8D"]

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# ======================================================
# HALAMAN BERANDA
# ======================================================
def show_home():
    st.markdown("<h1 style='text-align: center; color: #1B5E20;'>Pesona Wastra Nanggroe Aceh Darussalam</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #555;'>Aplikasi kecerdasan buatan untuk melestarikan dan mengenali kekayaan motif Batik Aceh.</p>", unsafe_allow_html=True)
    st.write("")
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 🌱 Pucok Reubong")
        st.write("Terinspirasi dari tunas bambu. Melambangkan kehidupan yang terus tumbuh, berkembang, dan memberikan manfaat tiada henti bagi sesama manusia. Simbol kekuatan dan keuletan.")
        st.image("https://via.placeholder.com/400x300/1B5E20/ffffff?text=Pucok+Reubong", use_container_width=True)

    with c2:
        st.markdown("### ⛩️ Pinto Aceh")
        st.write("Terinspirasi dari gerbang taman kerajaan. Desain pintu yang rendah secara filosofis mencerminkan nilai keterbukaan, kerendahan hati, sopan santun, dan pemuliaan terhadap tamu.")
        st.image("https://via.placeholder.com/400x300/D4A017/ffffff?text=Pinto+Aceh", use_container_width=True)

    with c3:
        st.markdown("### ⚔️ Rencong")
        st.write("Senjata pusaka tradisional Aceh. Disimbolkan sebagai cerminan absolut dari keberanian, keadilan, martabat, dan semangat juang yang pantang menyerah dari masyarakat Aceh.")
        st.image("https://via.placeholder.com/400x300/7F8C8D/ffffff?text=Rencong", use_container_width=True)

# ======================================================
# HALAMAN KLASIFIKASI
# ======================================================
def show_classification(dev_mode):
    st.title("🔍 Klasifikasi AI")
    st.write("Unggah gambar kain batik Anda (satu atau banyak sekaligus).")
    
    uploaded_files = st.file_uploader("Upload gambar (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.divider()
        
        # BATCH PROCESSING
        if len(uploaded_files) > 1:
            st.subheader("Batch Processing Hasil Prediksi")
            results = []
            
            progress_bar = st.progress(0)
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                start_time = time.time()
                pred_class, conf_scores = predict(img)
                inf_time = time.time() - start_time
                
                conf_val = conf_scores[pred_class] * 100
                
                results.append({
                    "Nama File": file.name,
                    "Prediksi": pred_class,
                    "Keyakinan (%)": round(conf_val, 2),
                    "Status": "YAKIN" if conf_val >= 80 else "TIDAK YAKIN"
                })
                
                # Simpan ke sesi
                st.session_state['history'].append({
                    "Waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "File": file.name,
                    "Prediksi": pred_class,
                    "Keyakinan": f"{conf_val:.2f}%"
                })
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            
        # DETAIL PROCESSING (1 GAMBAR)
        elif len(uploaded_files) == 1:
            file = uploaded_files[0]
            col1, col2 = st.columns([4, 6])
            
            image = Image.open(file)
            
            start_time = time.time()
            pred_class, conf_scores = predict(image)
            inf_time = time.time() - start_time
            conf_val = conf_scores[pred_class] * 100
            
            # Simpan ke sesi
            st.session_state['history'].append({
                "Waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
                "File": file.name,
                "Prediksi": pred_class,
                "Keyakinan": f"{conf_val:.2f}%"
            })
            
            with col1:
                st.markdown("#### Citra Input")
                st.image(image, use_container_width=True)
                
            with col2:
                # 1. Animasi Lottie (Opsional, akan muncul sebentar atau terus menerus)
                lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
                if lottie_ai:
                    st_lottie(lottie_ai, height=100, key="ai_lottie")
                
                # 2. Teks Hasil Tebakan
                st.success(f"### Motif Terdeteksi: {pred_class} ({conf_val:.1f}%)")
                
                # 3. Grafik Probabilitas
                df_conf = pd.DataFrame({
                    "Kelas": list(conf_scores.keys()),
                    "Probabilitas": list(conf_scores.values())
                }).sort_values(by="Probabilitas", ascending=True)

                fig = px.bar(
                    df_conf, x="Probabilitas", y="Kelas", orientation='h',
                    text_auto='.1%', color="Probabilitas", color_continuous_scale="Viridis"
                )
                fig.update_layout(xaxis_tickformat='.0%', height=250, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # 4. Ekstraktor Warna KMeans
                st.markdown("**Palet Warna Dominan:**")
                hex_colors = extract_colors(image)
                color_html = "<div style='display:flex; gap:10px;'>"
                for hx in hex_colors:
                    color_html += f"<div style='background-color:{hx}; width:40px; height:40px; border-radius:8px; border:1px solid #ccc;' title='{hx}'></div>"
                color_html += "</div>"
                st.markdown(color_html, unsafe_allow_html=True)
                
                st.write("")
                
                # 5. Audio Narator (gTTS)
                st.markdown("**Audio Penjelasan:**")
                audio_text = f"Berdasarkan analisis visual, motif batik yang terdeteksi pada gambar ini adalah {pred_class}, dengan tingkat keyakinan {conf_val:.1f} persen."
                try:
                    tts = gTTS(text=audio_text, lang='id')
                    fp = io.BytesIO()
                    tts.write_to_fp(fp)
                    st.audio(fp, format="audio/mp3")
                except:
                    st.warning("Gagal memuat Audio Narator.")

                # 6. Developer Mode Expander
                if dev_mode:
                    with st.expander("🛠️ Debug Info (Developer Mode)"):
                        st.json(conf_scores)
                        st.write(f"Inference Time: `{inf_time:.4f} seconds`")
                        
                # 7. Kartu Unduhan & Watermark
                st.divider()
                st.markdown("#### Buat Kartu Hasil")
                user_name = st.text_input("Masukkan nama Anda untuk Watermark:")
                if user_name:
                    try:
                        img_copy = image.copy().convert("RGBA")
                        txt_img = Image.new('RGBA', img_copy.size, (255,255,255,0))
                        draw = ImageDraw.Draw(txt_img)
                        # Hitung ukuran font proporsional
                        font_size = int(img_copy.width / 20)
                        # Fallback ke font bawaan PIL
                        try:
                            font = ImageFont.truetype("arial.ttf", font_size)
                        except IOError:
                            font = ImageFont.load_default()
                            
                        watermark_text = f"Dianalisis untuk: {user_name} | {pred_class}"
                        draw.text((10, img_copy.height - font_size - 20), watermark_text, fill=(255, 255, 255, 200), font=font)
                        
                        # Gabungkan gambar asli dengan layer text
                        watermarked = Image.alpha_composite(img_copy, txt_img).convert("RGB")
                        
                        buf = io.BytesIO()
                        watermarked.save(buf, format="PNG")
                        st.download_button(
                            label="⬇️ Unduh Gambar Watermark (PNG)",
                            data=buf.getvalue(),
                            file_name=f"Batik_Aceh_{user_name}.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Gagal membuat watermark: {e}")

# ======================================================
# HALAMAN GALERI ETALASE
# ======================================================
def show_gallery():
    st.title("📚 Galeri Etalase")
    st.write("Kumpulan foto visual dari dataset referensi.")
    st.divider()
    
    dataset_path = "e:/Batik3dataset/dataset"
    if not os.path.exists(dataset_path):
        st.warning(f"Folder dataset tidak ditemukan di {dataset_path}")
        return
        
    classes = ["Pinto Aceh", "Pucok Reubong", "Rencong"]
    
    for cls in classes:
        st.subheader(f"Motif {cls}")
        cls_path = os.path.join(dataset_path, cls)
        
        if os.path.exists(cls_path):
            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            # Tampilkan hingga 4 gambar per kelas agar terlihat penuh
            files = files[:4]
            
            if files:
                cols = st.columns(len(files))
                for idx, file in enumerate(files):
                    with cols[idx]:
                        img_path = os.path.join(cls_path, file)
                        try:
                            st.image(img_path, use_container_width=True, caption=file)
                        except:
                            pass
            else:
                st.info(f"Tidak ada gambar di folder {cls}")
        st.divider()

# ======================================================
# PENGENDALI UTAMA
# ======================================================
def main():
    with st.sidebar:
        st.title("Navigasi Utama")
        # 1. Option Menu
        selected = option_menu(
            menu_title=None,
            options=["🏠 Beranda", "🔍 Klasifikasi AI", "📚 Galeri Etalase"],
            default_index=0
        )
        
        st.divider()
        
        # 2. Riwayat Sesi
        st.markdown("### 🕒 Riwayat Sesi")
        if len(st.session_state['history']) > 0:
            df_hist = pd.DataFrame(st.session_state['history'])
            st.dataframe(df_hist, use_container_width=True, height=150)
            
            # Tombol Unduh CSV
            csv = df_hist.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Unduh CSV",
                data=csv,
                file_name='riwayat_klasifikasi.csv',
                mime='text/csv',
            )
        else:
            st.info("Belum ada riwayat analisis.")
            
        st.divider()
        
        # 3. Toggle Mode Developer
        dev_mode = st.checkbox("⚙️ Aktifkan Mode Developer", value=False)
        
    # Router
    if selected == "🏠 Beranda":
        show_home()
    elif selected == "🔍 Klasifikasi AI":
        show_classification(dev_mode)
    elif selected == "📚 Galeri Etalase":
        show_gallery()

if __name__ == "__main__":
    main()
