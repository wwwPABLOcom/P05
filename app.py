import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import joblib
import json
import os
import io
import tempfile
from PIL import Image
import matplotlib
matplotlib.use('Agg') # Evita errores de ventanas al dibujar espectrogramas en segundo plano
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- NUEVO IMPORT PARA YOUTUBE ---
import yt_dlp 

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
st.set_page_config(page_title="IA Clasificador Musical", page_icon="🎵")
st.title("🎵 Clasificador de Géneros Musicales")
st.write("Sube un archivo de audio o pega un enlace de YouTube para predecir su género.")

# Rutas de tus archivos (Deben estar en la misma carpeta que este app.py)
MODELO_PATH = "./clasificador_hibrido/Prueba/modelo_hibrido_definitivo.keras"
SCALER_PATH = "./clasificador_hibrido/Prueba/scaler_hibrido_definitivo.pkl"
CLASES_PATH = "./clasificador_hibrido/Prueba/clases_hibrido_definitivo.json"
N_CLASES = 10
IMG_SIZE = (128, 128)

# ==========================================
# 2. CARGA DE MODELO (En caché para que no cargue cada vez)
# ==========================================
@st.cache_resource
def load_models():
    modelo = tf.keras.models.load_model(MODELO_PATH)
    scalers = joblib.load(SCALER_PATH)
    with open(CLASES_PATH) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    return modelo, scalers, idx_to_class

# ==========================================
# 3. FUNCIONES DE EXTRACCIÓN (Tu IA)
# ==========================================
def extraer_features_wav(y: np.ndarray, sr: int) -> np.ndarray:
    feats = []
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats += [np.mean(chroma), np.var(chroma)]
    rms = librosa.feature.rms(y=y)
    feats += [np.mean(rms), np.var(rms)]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats += [np.mean(centroid), np.var(centroid)]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats += [np.mean(bandwidth), np.var(bandwidth)]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats += [np.mean(rolloff), np.var(rolloff)]
    zcr = librosa.feature.zero_crossing_rate(y)
    feats += [np.mean(zcr), np.var(zcr)]
    harmony, perceptr = librosa.effects.hpss(y)
    feats += [np.mean(harmony), np.var(harmony)]
    feats += [np.mean(perceptr), np.var(perceptr)]
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats += [float(np.atleast_1d(tempo)[0])]
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        feats += [np.mean(mfccs[i]), np.var(mfccs[i])]
    return np.array(feats, dtype=np.float32)

def audio_a_espectrograma(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(3, 3), dpi=72)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis('off')
    librosa.display.specshow(mel_db, sr=sr, fmax=8000, ax=ax, cmap='viridis')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB').resize(IMG_SIZE)
    img_arr = np.array(img, dtype=np.float32)
    return preprocess_input(img_arr)

# ==========================================
# 4. DESCARGA DESDE YOUTUBE (INTEGRADA)
# ==========================================
def descargar_audio_youtube(url):
    # Creamos una ruta segura en la carpeta temporal del sistema
    tmp_dir = tempfile.gettempdir()
    ruta_salida = os.path.join(tmp_dir, 'audio_yt_temporal.%(ext)s')
    ruta_final = os.path.join(tmp_dir, 'audio_yt_temporal.mp3')
    
    # Si existe un archivo anterior, lo borramos para evitar conflictos
    if os.path.exists(ruta_final):
        try:
            os.remove(ruta_final)
        except:
            pass

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': ruta_salida, 
        'quiet': True,
        'no_warnings': True
    }
    
    with st.spinner("Descargando audio de YouTube..."):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return ruta_final
        except Exception as e:
            st.error(f"❌ Error al descargar el video: {e}")
            return None

# ==========================================
# 5. LÓGICA DE PREDICCIÓN
# ==========================================
def analizar_audio(ruta_audio):
    modelo, scalers, idx_to_class = load_models()
    
    with st.spinner("Cargando matriz de audio con Librosa (esto puede tardar unos segundos)..."):
        y_full, sr = librosa.load(ruta_audio, mono=True)
        
    st.info("🎵 Analizando ventanas de sonido...")
    progress_bar = st.progress(0)
    
    ven_s = 20 * sr 
    salto_s = 10 * sr 
    inicios = list(range(0, max(1, len(y_full) - ven_s), salto_s))
    
    if len(inicios) == 0:
        st.warning("El audio es demasiado corto para las ventanas de tu modelo (requiere al menos 20 seg).")
        return None, None, None

    probas_acum = np.zeros(N_CLASES)
    
    for i, t_ini in enumerate(inicios):
        frag = y_full[t_ini : t_ini + ven_s]
        if len(frag) < ven_s: break 
        
        feats = extraer_features_wav(frag, sr).reshape(1, -1)
        csv_s = scalers['csv'].transform(feats)
        wav_s = scalers['wav'].transform(feats)
        img = audio_a_espectrograma(frag, sr)[np.newaxis]
        
        probas = modelo.predict([csv_s, img, wav_s], verbose=0)[0]
        probas_acum += probas
        
        # Animación de la barra de progreso
        progress_bar.progress((i + 1) / len(inicios))

    probas_finales = probas_acum / len(inicios)
    ranking = sorted(enumerate(probas_finales), key=lambda x: x[1], reverse=True)
    return ranking, idx_to_class, probas_finales

# ==========================================
# 6. INTERFAZ VISUAL (UI)
# ==========================================
metodo = st.radio("¿Qué método quieres usar?", ["Enlace de YouTube", "Subir archivo"])
ruta_a_procesar = None

if metodo == "Enlace de YouTube":
    youtube_url = st.text_input("Pega el enlace de YouTube:")
    
    if st.button("Analizar Enlace") and youtube_url:
        ruta_tmp = descargar_audio_youtube(youtube_url)
        if ruta_tmp:
            st.audio(ruta_tmp)
            ruta_a_procesar = ruta_tmp

elif metodo == "Subir archivo":
    archivo_subido = st.file_uploader("Sube tu canción (MP3, WAV)", type=["wav", "mp3"])
    if archivo_subido:
        st.audio(archivo_subido)
        if st.button("Analizar Archivo"):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_file.write(archivo_subido.getbuffer())
            tmp_file.close()
            ruta_a_procesar = tmp_file.name

# ==========================================
# 7. EJECUCIÓN DEL ANÁLISIS
# ==========================================
if ruta_a_procesar:
    st.markdown("---")
    st.subheader("Resultados del Modelo Híbrido")
    
    try:
        ranking, idx_to_class, probas_finales = analizar_audio(ruta_a_procesar)
        
        if ranking is not None:
            genero_1 = idx_to_class[ranking[0][0]].upper()
            prob_1 = ranking[0][1] * 100
            genero_2 = idx_to_class[ranking[1][0]].upper()
            prob_2 = ranking[1][1] * 100
            
            st.success("¡Análisis finalizado con éxito!")
            st.markdown(f"### 🏆 GÉNERO PRINCIPAL: **{genero_1}** ({prob_1:.1f}%)")
            st.markdown(f"#### 🥈 Sub-género: {genero_2} ({prob_2:.1f}%)")
            
            st.write("**Distribución de probabilidades por género:**")
            diccionario_grafico = {idx_to_class[i]: p for i, p in enumerate(probas_finales)}
            st.bar_chart(diccionario_grafico)
            
    except Exception as e:
        st.error(f"Error durante el análisis del audio: {str(e)}")
        
    finally:
        # Limpieza: Borrar el archivo temporal después de analizar para no ocupar espacio
        if os.path.exists(ruta_a_procesar):
            try:
                os.remove(ruta_a_procesar)
            except:
                pass
