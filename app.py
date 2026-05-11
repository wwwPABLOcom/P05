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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import yt_dlp

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
st.set_page_config(
    page_title="IA Clasificador Musical",
    page_icon="🎵",
    layout="centered"
)

# ==========================================
# CSS PERSONALIZADO
# ==========================================
st.markdown("""
<style>
    /* Fuente principal */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Fondo oscuro general */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #12102b 50%, #0d0d1a 100%);
        color: #e8e6f0;
    }

    /* Ocultar barra de menú y footer de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Contenedor principal */
    .block-container {
        padding: 2rem 2rem 4rem 2rem;
        max-width: 780px;
    }

    /* HEADER PERSONALIZADO */
    .hero-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem 1rem;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #7c3aed, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #9b97b8;
        font-weight: 400;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(124, 58, 237, 0.15);
        border: 1px solid rgba(124, 58, 237, 0.35);
        color: #c4b5fd;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 1rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    /* TARJETAS DE SECCIÓN */
    .section-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(167, 139, 250, 0.18);
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
    }

    /* RADIO BUTTONS (método de entrada) */
    .stRadio > label {
        color: #c4b5fd !important;
        font-weight: 500;
        font-size: 0.9rem;
        letter-spacing: 0.3px;
    }
    .stRadio div[role="radiogroup"] {
        display: flex;
        gap: 1rem;
        flex-direction: row !important;
    }
    .stRadio div[role="radiogroup"] label {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(167, 139, 250, 0.25);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        color: #d4d0ea !important;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.9rem;
    }
    .stRadio div[role="radiogroup"] label:hover {
        border-color: rgba(167, 139, 250, 0.6);
        background: rgba(124, 58, 237, 0.12);
    }

    /* INPUTS DE TEXTO */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(167, 139, 250, 0.3) !important;
        border-radius: 10px !important;
        color: #e8e6f0 !important;
        padding: 0.65rem 1rem !important;
        font-size: 0.9rem !important;
        transition: border-color 0.2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(167, 139, 250, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.15) !important;
    }
    .stTextInput > div > div > input::placeholder {
        color: #6b6880 !important;
    }

    /* BOTONES */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed, #6d28d9) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* FILE UPLOADER */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 2px dashed rgba(167, 139, 250, 0.3) !important;
        border-radius: 12px !important;
        transition: all 0.2s ease;
    }
    .stFileUploader > div:hover {
        border-color: rgba(167, 139, 250, 0.6) !important;
        background: rgba(124, 58, 237, 0.06) !important;
    }
    .stFileUploader label {
        color: #9b97b8 !important;
    }

    /* AUDIO PLAYER */
    .stAudio {
        margin: 0.75rem 0;
    }
    .stAudio audio {
        width: 100%;
        border-radius: 8px;
        filter: invert(10%) hue-rotate(240deg);
    }

    /* MENSAJES (info, warning, success, error) */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
    }
    div[data-testid="stNotification"] {
        border-radius: 10px !important;
    }

    /* SPINNER */
    .stSpinner > div {
        border-top-color: #7c3aed !important;
    }

    /* PROGRESS BAR */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7c3aed, #a78bfa) !important;
    }

    /* DIVIDER */
    hr {
        border: none;
        border-top: 1px solid rgba(167, 139, 250, 0.15);
        margin: 1.5rem 0;
    }

    /* RESULTADOS - tarjeta de resultado principal */
    .result-card {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(109, 40, 217, 0.08));
        border: 1px solid rgba(167, 139, 250, 0.35);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-genre {
        font-size: 2.8rem;
        font-weight: 700;
        color: #c4b5fd;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .result-percent {
        font-size: 1.1rem;
        color: #a78bfa;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    .result-secondary {
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #9b97b8;
    }
    .result-secondary span {
        color: #c4b5fd;
        font-weight: 600;
    }

    /* GRÁFICO DE BARRAS */
    .stVegaLiteChart, [data-testid="stArrowVegaLiteChart"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* SUBHEADER PERSONALIZADO */
    h2, h3 {
        color: #d4d0ea !important;
        font-weight: 600 !important;
    }

    /* Labels */
    label, .stMarkdown p {
        color: #c0bdd6 !important;
    }

    /* Info box de advertencia */
    .stWarning {
        background: rgba(251, 191, 36, 0.08) !important;
        border-left: 3px solid #f59e0b !important;
        color: #fcd34d !important;
        border-radius: 0 8px 8px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER PERSONALIZADO
# ==========================================
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">🤖 Modelo Híbrido · Deep Learning</div>
    <div class="hero-title">🎵 Clasificador de Géneros</div>
    <div class="hero-subtitle">
        Inteligencia artificial que analiza el audio de una canción<br>
        y detecta su género musical con alta precisión.
    </div>
</div>
""", unsafe_allow_html=True)

# Rutas de archivos
MODELO_PATH = "./clasificador_hibrido/Prueba/modelo_hibrido_definitivo.keras"
SCALER_PATH = "./clasificador_hibrido/Prueba/scaler_hibrido_definitivo.pkl"
CLASES_PATH = "./clasificador_hibrido/Prueba/clases_hibrido_definitivo.json"
N_CLASES = 10
IMG_SIZE = (128, 128)

# ==========================================
# 2. CARGA DE MODELO
# ==========================================
@st.cache_resource
def load_models():
    modelo = tf.keras.models.load_model(MODELO_PATH)
    scalers = joblib.load(SCALER_PATH)
    with open(CLASES_PATH) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    return modelo, scalers, idx_to_class

# ==========================================
# 3. FUNCIONES DE EXTRACCIÓN
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
# 4. DESCARGA DESDE YOUTUBE
# ==========================================
def descargar_audio_youtube(url):
    tmp_dir = tempfile.gettempdir()
    ruta_salida = os.path.join(tmp_dir, 'audio_yt_temporal.%(ext)s')
    ruta_final = os.path.join(tmp_dir, 'audio_yt_temporal.mp3')
    if os.path.exists(ruta_final):
        try:
            os.remove(ruta_final)
        except:
            pass
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': ruta_salida,
        'quiet': True,
        'no_warnings': True
    }
    with st.spinner("⬇️ Descargando audio de YouTube..."):
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
    with st.spinner("🔊 Cargando audio..."):
        y_full, sr = librosa.load(ruta_audio, mono=True)
    st.info("🎵 Analizando ventanas de sonido...")
    progress_bar = st.progress(0)
    ven_s = 20 * sr
    salto_s = 10 * sr
    inicios = list(range(0, max(1, len(y_full) - ven_s), salto_s))
    if len(inicios) == 0:
        st.warning("⚠️ El audio es demasiado corto (requiere al menos 20 segundos).")
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
        progress_bar.progress((i + 1) / len(inicios))
    probas_finales = probas_acum / len(inicios)
    ranking = sorted(enumerate(probas_finales), key=lambda x: x[1], reverse=True)
    return ranking, idx_to_class, probas_finales

# ==========================================
# 6. INTERFAZ DE ENTRADA
# ==========================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)

metodo = st.radio("Método de entrada", ["🔗 Enlace de YouTube", "📁 Subir archivo"])
ruta_a_procesar = None

if "YouTube" in metodo:
    youtube_url = st.text_input("URL de YouTube", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("🎵 Analizar enlace") and youtube_url:
        ruta_tmp = descargar_audio_youtube(youtube_url)
        if ruta_tmp:
            st.audio(ruta_tmp)
            ruta_a_procesar = ruta_tmp
    st.warning("⚠️ Las descargas desde YouTube pueden estar bloqueadas en entornos web. Si falla, usa la opción **Subir archivo**.")

elif "Subir" in metodo:
    archivo_subido = st.file_uploader("Arrastra o selecciona tu canción", type=["wav", "mp3"])
    if archivo_subido:
        st.audio(archivo_subido)
        if st.button("🎵 Analizar archivo"):
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_file.write(archivo_subido.getbuffer())
            tmp_file.close()
            ruta_a_procesar = tmp_file.name

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 7. RESULTADOS
# ==========================================
if ruta_a_procesar:
    try:
        ranking, idx_to_class, probas_finales = analizar_audio(ruta_a_procesar)
        if ranking is not None:
            genero_1 = idx_to_class[ranking[0][0]].upper()
            prob_1 = ranking[0][1] * 100
            genero_2 = idx_to_class[ranking[1][0]].upper()
            prob_2 = ranking[1][1] * 100

            st.success("✅ ¡Análisis completado!")

            st.markdown(f"""
            <div class="result-card">
                <div style="font-size:0.85rem; color:#9b97b8; font-weight:500; text-transform:uppercase;
                            letter-spacing:1px; margin-bottom:0.5rem;">Género principal detectado</div>
                <div class="result-genre">🎶 {genero_1}</div>
                <div class="result-percent">{prob_1:.1f}% de confianza</div>
                <div class="result-secondary">
                    🥈 Segundo candidato: <span>{genero_2}</span> ({prob_2:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### Distribución de probabilidades por género")
            diccionario_grafico = {idx_to_class[i]: float(p) for i, p in enumerate(probas_finales)}
            st.bar_chart(diccionario_grafico, color="#7c3aed")

    except Exception as e:
        st.error(f"❌ Error durante el análisis: {str(e)}")
    finally:
        if os.path.exists(ruta_a_procesar):
            try:
                os.remove(ruta_a_procesar)
            except:
                pass

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div style="text-align:center; margin-top:3rem; padding-top:1.5rem;
            border-top:1px solid rgba(167,139,250,0.12);
            color:#4a4760; font-size:0.8rem;">
    Modelo Híbrido · CNN + MFCCs · Entrenado con GTZAN Dataset
</div>
""", unsafe_allow_html=True)
