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
# 1. CONFIGURACIÓN
# ==========================================
st.set_page_config(
    page_title="IA Clasificador Musical",
    page_icon="🎵",
    layout="centered"
)

# ==========================================
# CSS COMPLETO
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* === FONDO PRINCIPAL === */
    .stApp {
        background-color: #07071a;
        background-image:
            radial-gradient(ellipse 80% 50% at 20% -10%, rgba(120, 60, 255, 0.25) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 110%, rgba(30, 180, 255, 0.12) 0%, transparent 55%),
            radial-gradient(ellipse 50% 50% at 50% 50%, rgba(180, 60, 120, 0.06) 0%, transparent 70%);
        min-height: 100vh;
        color: #e8e6f0;
    }

    /* Fondo con cuadrícula sutil */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image:
            linear-gradient(rgba(120, 60, 255, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(120, 60, 255, 0.04) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events: none;
        z-index: 0;
    }

    /* === OCULTAR ELEMENTOS DE STREAMLIT === */
    #MainMenu, footer, header { visibility: hidden; }

    /* === CONTENEDOR PRINCIPAL === */
    .block-container {
        padding: 1.5rem 2rem 5rem 2rem;
        max-width: 800px;
        position: relative;
        z-index: 1;
    }

    /* === HERO / HEADER === */
    .hero-wrap {
        text-align: center;
        padding: 3rem 1rem 2.5rem 1rem;
        position: relative;
    }

    /* Vinilo decorativo en el fondo del header */
    .vinyl-bg {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 320px;
        height: 320px;
        border-radius: 50%;
        background: conic-gradient(
            from 0deg,
            rgba(120, 60, 255, 0.06),
            rgba(180, 80, 255, 0.03),
            rgba(60, 180, 255, 0.06),
            rgba(120, 60, 255, 0.06)
        );
        animation: spin 20s linear infinite;
        pointer-events: none;
        z-index: 0;
    }
    .vinyl-bg::after {
        content: '';
        position: absolute;
        inset: 40px;
        border-radius: 50%;
        background: conic-gradient(
            from 90deg,
            rgba(255,80,180,0.05),
            rgba(120,60,255,0.08),
            rgba(255,80,180,0.05)
        );
        animation: spin 12s linear infinite reverse;
    }

    @keyframes spin {
        to { transform: translate(-50%, -50%) rotate(360deg); }
    }

    /* Ondas de audio debajo del header */
    .audio-waves {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
        height: 40px;
        margin: 0.75rem 0 0 0;
    }
    .wave-bar {
        width: 3px;
        border-radius: 3px;
        background: linear-gradient(to top, #7c3aed, #a78bfa);
        animation: wave-anim var(--dur, 1s) ease-in-out infinite alternate;
        transform-origin: bottom;
    }
    @keyframes wave-anim {
        from { transform: scaleY(0.2); opacity: 0.4; }
        to   { transform: scaleY(1);   opacity: 1;   }
    }

    .hero-badge {
        display: inline-block;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(167, 139, 250, 0.3);
        color: #c4b5fd;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 5px 14px;
        border-radius: 999px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 1.2rem;
        position: relative;
        z-index: 1;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #e0d7ff 10%, #a78bfa 45%, #7c3aed 70%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1.5px;
        line-height: 1.1;
        margin-bottom: 0.6rem;
        position: relative;
        z-index: 1;
    }
    .hero-sub {
        font-size: 1rem;
        color: #8b88a8;
        font-weight: 400;
        line-height: 1.6;
        max-width: 480px;
        margin: 0 auto;
        position: relative;
        z-index: 1;
    }

    /* === TARJETAS GLASSMORPHISM === */
    .glass-card {
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(167, 139, 250, 0.14);
        border-radius: 20px;
        padding: 1.75rem 2rem;
        margin-bottom: 1.25rem;
        position: relative;
        overflow: hidden;
    }
    /* Brillo superior en la tarjeta */
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.4), transparent);
    }

    /* === RADIO BUTTONS === */
    .stRadio > label {
        color: #9b97b8 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        margin-bottom: 0.75rem;
        display: block;
    }
    .stRadio div[role="radiogroup"] {
        display: flex !important;
        gap: 0.75rem !important;
        flex-direction: row !important;
    }
    .stRadio div[role="radiogroup"] label {
        flex: 1;
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(167, 139, 250, 0.2) !important;
        border-radius: 12px !important;
        padding: 0.75rem 1rem !important;
        color: #c4b5fd !important;
        cursor: pointer;
        transition: all 0.25s ease;
        text-align: center;
        font-size: 0.9rem !important;
    }
    .stRadio div[role="radiogroup"] label:hover {
        border-color: rgba(167, 139, 250, 0.55) !important;
        background: rgba(124, 58, 237, 0.1) !important;
    }

    /* === INPUT DE TEXTO === */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(167, 139, 250, 0.22) !important;
        border-radius: 12px !important;
        color: #e8e6f0 !important;
        padding: 0.7rem 1.1rem !important;
        font-size: 0.9rem !important;
        transition: all 0.2s;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(167, 139, 250, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.12), 0 0 20px rgba(124, 58, 237, 0.08) !important;
        background: rgba(255,255,255,0.07) !important;
    }
    .stTextInput > div > div > input::placeholder { color: #4f4c6b !important; }

    /* === BOTONES === */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 50%, #5b21b6 100%) !important;
        color: #f3f0ff !important;
        border: 1px solid rgba(167,139,250,0.3) !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        letter-spacing: 0.4px !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 20px rgba(124,58,237,0.35), inset 0 1px 0 rgba(255,255,255,0.1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0; left: -100%;
        width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.07), transparent);
        transition: left 0.4s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 50%, #6d28d9 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(124,58,237,0.45), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    }
    .stButton > button:hover::before { left: 100%; }
    .stButton > button:active { transform: translateY(0px) !important; }

    /* === FILE UPLOADER === */
    .stFileUploader {
        background: rgba(124, 58, 237, 0.04) !important;
        border: 2px dashed rgba(167, 139, 250, 0.22) !important;
        border-radius: 16px !important;
        padding: 0.5rem;
        transition: all 0.2s;
    }
    .stFileUploader:hover {
        border-color: rgba(167, 139, 250, 0.5) !important;
        background: rgba(124, 58, 237, 0.08) !important;
    }
    .stFileUploader section { background: transparent !important; }
    .stFileUploader label { color: #8b88a8 !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] small { color: #5f5d7a !important; }

    /* === AUDIO PLAYER === */
    .stAudio { margin: 0.75rem 0; }
    audio {
        width: 100%;
        border-radius: 10px;
    }

    /* === MENSAJES === */
    .stAlert, div[data-testid="stNotification"] {
        border-radius: 12px !important;
        backdrop-filter: blur(8px) !important;
    }
    div[data-testid="stNotificationContentWarning"] {
        background: rgba(245, 158, 11, 0.08) !important;
        border: 1px solid rgba(245, 158, 11, 0.25) !important;
        color: #fcd34d !important;
    }
    div[data-testid="stNotificationContentSuccess"] {
        background: rgba(16, 185, 129, 0.08) !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        color: #6ee7b7 !important;
    }
    div[data-testid="stNotificationContentInfo"] {
        background: rgba(124, 58, 237, 0.1) !important;
        border: 1px solid rgba(167, 139, 250, 0.25) !important;
        color: #c4b5fd !important;
    }

    /* === SPINNER === */
    .stSpinner > div { border-top-color: #7c3aed !important; }

    /* === PROGRESS BAR === */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #5b21b6, #7c3aed, #a78bfa) !important;
        border-radius: 999px !important;
        transition: width 0.3s ease;
        box-shadow: 0 0 12px rgba(124,58,237,0.5) !important;
    }
    .stProgress > div > div {
        background: rgba(124,58,237,0.12) !important;
        border-radius: 999px !important;
    }

    /* === DIVIDER === */
    hr {
        border: none;
        border-top: 1px solid rgba(167,139,250,0.1);
        margin: 2rem 0;
    }

    /* === TARJETA DE RESULTADO PRINCIPAL === */
    .result-main {
        position: relative;
        padding: 2.5rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1.25rem 0;
        overflow: hidden;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(167,139,250,0.2);
    }
    /* Orbe de brillo detrás del resultado */
    .result-main::before {
        content: '';
        position: absolute;
        width: 300px; height: 300px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(124,58,237,0.2) 0%, transparent 70%);
        top: 50%; left: 50%;
        transform: translate(-50%,-50%);
        pointer-events: none;
    }
    .result-crown { font-size: 2rem; margin-bottom: 0.25rem; }
    .result-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6b68a0;
        margin-bottom: 0.5rem;
    }
    .result-genre {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #f0eaff, #c4b5fd, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -2px;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .result-conf {
        display: inline-block;
        font-size: 1rem;
        font-weight: 500;
        color: #8b88a8;
        background: rgba(124,58,237,0.1);
        border: 1px solid rgba(167,139,250,0.2);
        border-radius: 999px;
        padding: 4px 16px;
        margin-bottom: 1.25rem;
    }
    .result-conf span { color: #c4b5fd; font-weight: 700; }
    .result-runner {
        font-size: 0.875rem;
        color: #6b68a0;
        margin-top: 0.5rem;
    }
    .result-runner b { color: #9b97b8; font-weight: 600; }

    /* Separador decorativo con nota musical */
    .music-divider {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 1.5rem 0;
        color: rgba(167,139,250,0.3);
        font-size: 0.75rem;
        letter-spacing: 2px;
        text-transform: uppercase;
    }
    .music-divider::before, .music-divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(167,139,250,0.2), transparent);
    }

    /* Chip de género para el ranking */
    .genre-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(167,139,250,0.15);
        border-radius: 10px;
        padding: 8px 14px;
        margin: 4px;
        font-size: 0.85rem;
        color: #c4b5fd;
    }
    .genre-chip .pct {
        color: #7c3aed;
        font-weight: 700;
        font-size: 0.78rem;
        font-variant-numeric: tabular-nums;
    }

    /* Labels y texto */
    label, .stMarkdown p { color: #c0bdd6 !important; }
    h2, h3 { color: #d4d0ea !important; font-weight: 600 !important; }
    h3 { font-size: 1rem !important; }

    /* Barra de gráfico */
    .stVegaLiteChart canvas, .stVegaLiteChart svg { border-radius: 8px; }

    /* Footer */
    .site-footer {
        text-align: center;
        margin-top: 4rem;
        padding-top: 2rem;
        border-top: 1px solid rgba(167,139,250,0.08);
        color: #3d3b57;
        font-size: 0.78rem;
        letter-spacing: 0.5px;
    }
    .site-footer span { color: #5b5880; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown("""
<div class="hero-wrap">
    <div class="vinyl-bg"></div>
    <div class="hero-badge">🤖 Modelo Híbrido · CNN + MFCCs</div>
    <div class="hero-title">Clasificador de Géneros</div>
    <div class="hero-sub">
        Sube una canción o pega un enlace de YouTube.<br>
        La IA analiza el espectrograma y las características acústicas para detectar el género.
    </div>
    <div class="audio-waves">
        <div class="wave-bar" style="height:28px; --dur:0.7s;"></div>
        <div class="wave-bar" style="height:36px; --dur:0.9s;"></div>
        <div class="wave-bar" style="height:20px; --dur:0.6s;"></div>
        <div class="wave-bar" style="height:44px; --dur:1.1s;"></div>
        <div class="wave-bar" style="height:32px; --dur:0.8s;"></div>
        <div class="wave-bar" style="height:48px; --dur:0.65s;"></div>
        <div class="wave-bar" style="height:28px; --dur:1.0s;"></div>
        <div class="wave-bar" style="height:40px; --dur:0.75s;"></div>
        <div class="wave-bar" style="height:18px; --dur:0.55s;"></div>
        <div class="wave-bar" style="height:36px; --dur:0.9s;"></div>
        <div class="wave-bar" style="height:26px; --dur:0.7s;"></div>
        <div class="wave-bar" style="height:42px; --dur:1.05s;"></div>
        <div class="wave-bar" style="height:30px; --dur:0.8s;"></div>
        <div class="wave-bar" style="height:22px; --dur:0.6s;"></div>
        <div class="wave-bar" style="height:38px; --dur:0.95s;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# RUTAS Y CONFIG
# ==========================================
MODELO_PATH = "./clasificador_hibrido/Prueba/modelo_hibrido_definitivo.keras"
SCALER_PATH = "./clasificador_hibrido/Prueba/scaler_hibrido_definitivo.pkl"
CLASES_PATH = "./clasificador_hibrido/Prueba/clases_hibrido_definitivo.json"
N_CLASES = 10
IMG_SIZE = (128, 128)

# Colores por género (para dar personalidad al resultado)
GENRE_COLORS = {
    "blues":     {"bg": "rgba(30,100,255,0.12)",  "border": "rgba(100,160,255,0.35)", "text": "#93c5fd"},
    "classical": {"bg": "rgba(200,150,80,0.12)",  "border": "rgba(220,180,100,0.35)", "text": "#fde68a"},
    "country":   {"bg": "rgba(180,120,40,0.12)",  "border": "rgba(210,160,80,0.35)",  "text": "#fcd34d"},
    "disco":     {"bg": "rgba(220,60,180,0.12)",  "border": "rgba(240,100,200,0.35)", "text": "#f9a8d4"},
    "hiphop":    {"bg": "rgba(255,80,50,0.12)",   "border": "rgba(255,120,80,0.35)",  "text": "#fca5a5"},
    "jazz":      {"bg": "rgba(60,180,180,0.12)",  "border": "rgba(80,210,210,0.35)",  "text": "#5eead4"},
    "metal":     {"bg": "rgba(100,100,120,0.12)", "border": "rgba(150,150,180,0.35)", "text": "#c4c4d4"},
    "pop":       {"bg": "rgba(200,60,220,0.12)",  "border": "rgba(220,100,240,0.35)", "text": "#e879f9"},
    "reggae":    {"bg": "rgba(30,180,80,0.12)",   "border": "rgba(60,220,100,0.35)",  "text": "#6ee7b7"},
    "rock":      {"bg": "rgba(220,80,40,0.12)",   "border": "rgba(240,120,60,0.35)",  "text": "#fb923c"},
}
DEFAULT_COLOR = {"bg": "rgba(124,58,237,0.12)", "border": "rgba(167,139,250,0.35)", "text": "#c4b5fd"}

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
    feats += [np.mean(harmony), np.var(harmony), np.mean(perceptr), np.var(perceptr)]
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
# 4. DESCARGA YOUTUBE
# ==========================================
def descargar_audio_youtube(url):
    tmp_dir = tempfile.gettempdir()
    ruta_salida = os.path.join(tmp_dir, 'audio_yt_temporal.%(ext)s')
    ruta_final  = os.path.join(tmp_dir, 'audio_yt_temporal.mp3')
    if os.path.exists(ruta_final):
        try: os.remove(ruta_final)
        except: pass
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': ruta_salida, 'quiet': True, 'no_warnings': True
    }
    with st.spinner("⬇️ Descargando audio de YouTube..."):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return ruta_final
        except Exception as e:
            st.error(f"❌ Error al descargar: {e}")
            return None

# ==========================================
# 5. PREDICCIÓN
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
        st.warning("⚠️ Audio demasiado corto (mínimo 20 segundos).")
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
# 6. UI DE ENTRADA
# ==========================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
metodo = st.radio("Método de entrada", ["🔗 Enlace de YouTube", "📁 Subir archivo"])
ruta_a_procesar = None

if "YouTube" in metodo:
    youtube_url = st.text_input("", placeholder="https://www.youtube.com/watch?v=...")
    if st.button("🎵 Analizar enlace") and youtube_url:
        ruta_tmp = descargar_audio_youtube(youtube_url)
        if ruta_tmp:
            st.audio(ruta_tmp)
            ruta_a_procesar = ruta_tmp
    st.warning("⚠️ Las descargas de YouTube pueden bloquearse en entornos web. Si falla, usa **Subir archivo**.")

elif "Subir" in metodo:
    archivo_subido = st.file_uploader("", type=["wav", "mp3"], label_visibility="collapsed")
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
            genero_key = idx_to_class[ranking[0][0]].lower()
            genero_1   = genero_key.upper()
            prob_1     = ranking[0][1] * 100
            genero_2   = idx_to_class[ranking[1][0]]
            prob_2     = ranking[1][1] * 100
            colores    = GENRE_COLORS.get(genero_key, DEFAULT_COLOR)

            st.success("✅ Análisis completado")

            # Tarjeta de resultado con colores únicos por género
            st.markdown(f"""
            <div class="result-main" style="
                background: {colores['bg']};
                border-color: {colores['border']};
            ">
                <div class="result-crown">🎶</div>
                <div class="result-label">Género principal detectado</div>
                <div class="result-genre" style="
                    background: linear-gradient(135deg, #ffffff, {colores['text']});
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">{genero_1}</div>
                <div class="result-conf">Confianza: <span>{prob_1:.1f}%</span></div>
                <div class="result-runner">🥈 Segundo candidato: <b>{genero_2.upper()}</b> — {prob_2:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Separador decorativo
            st.markdown('<div class="music-divider">♪ distribución completa ♪</div>', unsafe_allow_html=True)

            # Chips del top 5
            chips_html = '<div style="text-align:center; margin-bottom:1rem;">'
            for idx_c, prob_c in ranking[:5]:
                nombre = idx_to_class[idx_c]
                c = GENRE_COLORS.get(nombre.lower(), DEFAULT_COLOR)
                chips_html += f'''<span class="genre-chip" style="border-color:{c['border']}; color:{c['text']};">
                    {nombre.capitalize()} <span class="pct">{prob_c*100:.1f}%</span>
                </span>'''
            chips_html += '</div>'
            st.markdown(chips_html, unsafe_allow_html=True)

            # Gráfico de barras
            diccionario_grafico = {idx_to_class[i]: float(p) for i, p in enumerate(probas_finales)}
            st.bar_chart(diccionario_grafico, color="#7c3aed")

    except Exception as e:
        st.error(f"❌ Error durante el análisis: {str(e)}")
    finally:
        if os.path.exists(ruta_a_procesar):
            try: os.remove(ruta_a_procesar)
            except: pass

# ==========================================
# FOOTER
# ==========================================
st.markdown("""
<div class="site-footer">
    <span>CNN + MFCCs · Entrenado con GTZAN Dataset · 10 géneros</span><br>
    Clasificador de géneros musicales · IA
</div>
""", unsafe_allow_html=True)
