"""
╔══════════════════════════════════════════════════════════════════════╗
║          CLASIFICADOR HÍBRIDO DE GÉNEROS MUSICALES                   ║
║     CSV features  +  Espectrogramas PNG  +  Audio WAV directo        ║
╚══════════════════════════════════════════════════════════════════════╝

ARQUITECTURA — 3 ramas que se fusionan:

  [WAV audio] ──→ Extracción librosa ──→ 57 features
                                              │
  [PNG imagen] ──→ MobileNetV2 (CNN) ──→ 128 features   ──→ FUSIÓN ──→ Softmax(10)
                                              │                  │
  [CSV features] ──→ Normalización ──→ 57 features             Dense(256)

  Por qué esto es mejor que cada rama sola:
  - El CSV captura estadísticas globales precisas
  - La CNN captura patrones visuales del espectrograma
  - El WAV directo aporta features calculadas en tiempo real
    con los mismos parámetros para cualquier canción nueva
  - La fusión permite al modelo combinar evidencias de los 3

ESTRUCTURA DE ARCHIVOS NECESARIA:
    features_3_sec.csv          ← tu CSV original
    images_original/            ← descomprime images_original.zip
        blues/  classical/  country/  disco/  hiphop/
        jazz/   metal/      pop/      reggae/  rock/
    genres_original/            ← descomprime los 3 zips aquí
        blues/  classical/  country/  disco/  hiphop/
        jazz/   metal/      pop/      reggae/  rock/

REQUISITOS:
    pip install tensorflow librosa pillow numpy pandas scikit-learn matplotlib joblib

USO:
    python clasificador_hibrido.py                    # entrena
    python clasificador_hibrido.py mi_cancion.wav     # predice
    python clasificador_hibrido.py mi_cancion.mp3     # también mp3
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import regularizers

import os
import sys
import json
import zipfile
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────
CSV_PATH      = "features_30_sec.csv"
IMAGES_DIR    = "images_original"
WAVS_DIR      = "genres_original"
MODELO_PATH   = "modelo_hibrido2(83%)_eliminar.keras"
SCALER_PATH   = "scaler_hibrido2(83%)_eliminar.pkl"
CLASES_PATH   = "clases_hibrido2(83%)_eliminar.json"

IMG_SIZE      = (96, 96)    # Resolución nativa óptima para MobileNetV2   #Antes: (128, 128) # Más pequeño que antes → más rápido, suficiente
BATCH_SIZE    = 32 #Antes 16
EPOCHS_FASE1  = 40
EPOCHS_FASE2  = 10
GENEROS       = sorted(['blues','classical','country','disco','hiphop',
                         'jazz','metal','pop','reggae','rock'])
N_CLASES      = 10
N_FEATURES    = 57            # Columnas numéricas del CSV


# ══════════════════════════════════════════════════════════════════
# PARTE 1: EXTRACCIÓN DE FEATURES DE AUDIO
#   Misma función usada tanto para entrenar como para predecir
#   → garantiza consistencia entre entrenamiento y predicción
# ══════════════════════════════════════════════════════════════════

def extraer_features_wav(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extrae 57 features de audio de un fragmento numpy.
    Exactamente las mismas features que tiene el CSV de GTZAN.
    """
    import librosa

    feats = []

    # Chroma STFT — armonía (12 notas cromáticas)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    feats += [np.mean(chroma), np.var(chroma)]

    # RMS — energía / volumen
    rms = librosa.feature.rms(y=y)
    feats += [np.mean(rms), np.var(rms)]

    # Spectral Centroid — brillo del sonido
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats += [np.mean(centroid), np.var(centroid)]

    # Spectral Bandwidth — anchura espectral
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats += [np.mean(bandwidth), np.var(bandwidth)]

    # Spectral Rolloff — frecuencia de corte energético
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats += [np.mean(rolloff), np.var(rolloff)]

    # Zero Crossing Rate — ritmo/percusión
    zcr = librosa.feature.zero_crossing_rate(y)
    feats += [np.mean(zcr), np.var(zcr)]

    # Harmony y Perceptr — componentes armónica y percusiva
    harmony, perceptr = librosa.effects.hpss(y)
    feats += [np.mean(harmony), np.var(harmony)]
    feats += [np.mean(perceptr), np.var(perceptr)]

    # Tempo — BPM estimado
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats += [float(np.atleast_1d(tempo)[0])]

    # MFCCs 1–20 (media y varianza) — timbre y textura
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        feats += [np.mean(mfccs[i]), np.var(mfccs[i])]

    return np.array(feats, dtype=np.float32)  # shape (57,)


def audio_a_espectrograma(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa.display
    from PIL import Image
    import io
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

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
    
    # IMPORTANTE: Usamos el preprocesamiento nativo de MobileNetV2
    return preprocess_input(img_arr)

# ══════════════════════════════════════════════════════════════════
# PARTE 2: CARGA Y PREPARACIÓN DEL DATASET COMPLETO
# ══════════════════════════════════════════════════════════════════

def cargar_dataset_completo():
    """
    Construye el dataset fusionando las 3 fuentes a nivel de canción:
      - CSV: promedia los 10 fragmentos de 3s → 1 vector de 57 features por canción
      - PNG: carga la imagen del espectrograma correspondiente
      - WAV: extrae features directamente del audio completo

    Devuelve:
      X_csv   : (N, 57)        — features numéricas
      X_img   : (N, H, W, 3)   — espectrogramas como imágenes
      X_wav   : (N, 57)        — features extraídas del wav directo
      y       : (N,)            — etiquetas numéricas
      generos : lista de nombres de género
    """
    import librosa
    from PIL import Image

    print("=" * 60)
    print("  CARGANDO DATASET HÍBRIDO (CSV + PNG + WAV)")
    print("=" * 60)

    # ── 1. Cargar CSV y promediar fragmentos por canción ──────────
    print("\n  [1/3] Cargando CSV...")
    df = pd.read_csv(CSV_PATH)
    feat_cols = [c for c in df.columns if c not in ['filename','length','label']]

    # Extraer número de canción base: "blues.00000.3.wav" → "blues.00000"
    df['cancion_base'] = df['filename'].str.extract(r'(\w+\.\d+)\.\d+\.wav')[0]

    # Promediar los ~10 fragmentos de 3s de cada canción
    df_cancion = df.groupby(['cancion_base','label'])[feat_cols].mean().reset_index()
    print(f"     {len(df_cancion)} canciones únicas tras agrupar fragmentos")

    # ── 2. Cargar imágenes PNG ────────────────────────────────────
    print("\n  [2/3] Cargando imágenes PNG...")
    X_csv_list, X_img_list, X_wav_list, y_list = [], [], [], []

    genero_a_idx = {g: i for i, g in enumerate(GENEROS)}
    errores = 0

    for _, fila in df_cancion.iterrows():
        genero      = fila['label']
        cancion_id  = fila['cancion_base']          # ej: "blues.00000"
        numero      = cancion_id.split('.')[1]       # ej: "00000"

        # Ruta imagen: images_original/blues/blues00000.png
        ruta_img = os.path.join(IMAGES_DIR, genero, f"{genero}{numero}.png")

        # Ruta wav: genres_original/blues/blues.00000.wav
        ruta_wav = os.path.join(WAVS_DIR, genero, f"{cancion_id}.wav")

        # Saltar si falta algún archivo
        if not os.path.exists(ruta_img) or not os.path.exists(ruta_wav):
            errores += 1
            continue

        # Cargar imagen
        # Cargar imagen y aplicar preprocess_input correcto
        img = Image.open(ruta_img).convert('RGB').resize(IMG_SIZE)
        img_arr = np.array(img, dtype=np.float32)
        img_arr = preprocess_input(img_arr)

        # Cargar wav y extraer features
        try:
            y_audio, sr = librosa.load(ruta_wav, mono=True, duration=30)
            wav_feats = extraer_features_wav(y_audio, sr)
        except Exception:
            errores += 1
            continue

        X_csv_list.append(fila[feat_cols].values.astype(np.float32))
        X_img_list.append(img_arr)
        X_wav_list.append(wav_feats)
        y_list.append(genero_a_idx[genero])

    print(f"     Cargadas: {len(X_csv_list)} canciones  |  Errores/omitidas: {errores}")

    X_csv = np.stack(X_csv_list)   # (N, 57)
    X_img = np.stack(X_img_list)   # (N, H, W, 3)
    X_wav = np.stack(X_wav_list)   # (N, 57)
    y     = np.array(y_list)       # (N,)

    print(f"\n  Shapes finales:")
    print(f"     X_csv: {X_csv.shape}")
    print(f"     X_img: {X_img.shape}")
    print(f"     X_wav: {X_wav.shape}")
    print(f"     y:     {y.shape}")

    return X_csv, X_img, X_wav, y


# ══════════════════════════════════════════════════════════════════
# PARTE 3: CONSTRUCCIÓN DEL MODELO HÍBRIDO
# ══════════════════════════════════════════════════════════════════
def construir_modelo_hibrido():
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import regularizers

    print("\n" + "=" * 60)
    print("  CONSTRUYENDO MODELO HÍBRIDO OPTIMIZADO (Buscando >80%)")
    print("=" * 60)

    # ── Rama A: CSV features ──────────────────────────────────────
    inp_csv = layers.Input(shape=(N_FEATURES,), name='input_csv')
    x_csv = layers.GaussianNoise(0.01)(inp_csv) 
    # Freno relajado a 0.001
    x_csv = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_csv) 
    x_csv = layers.BatchNormalization()(x_csv)
    x_csv = layers.Dropout(0.4)(x_csv)
    x_csv = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_csv)
    x_csv = layers.BatchNormalization()(x_csv)

    # ── Rama B: Imagen espectrograma (El motor principal) ─────────
    inp_img = layers.Input(shape=(128, 128, 3), name='input_img')
    
    x_img_aug = layers.RandomTranslation(height_factor=0.0, width_factor=0.05)(inp_img)

    base_cnn = MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet'
    )
    base_cnn.trainable = False 
    
    x_img = base_cnn(x_img_aug, training=False) 
    x_img = layers.GlobalAveragePooling2D()(x_img)
    
    # IMPORTANTE: Subimos a 256 para que la imagen tenga más peso en la decisión final
    x_img = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dropout(0.5)(x_img)

    # ── Rama C: WAV features ──────────────────────────────────────
    inp_wav = layers.Input(shape=(N_FEATURES,), name='input_wav')
    x_wav = layers.GaussianNoise(0.01)(inp_wav) 
    # Freno relajado a 0.001
    x_wav = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_wav)
    x_wav = layers.BatchNormalization()(x_wav)
    x_wav = layers.Dropout(0.4)(x_wav)
    x_wav = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_wav)
    x_wav = layers.BatchNormalization()(x_wav)

    # ── Fusión de las 3 ramas ─────────────────────────────────────
    fusion = layers.Concatenate(name='fusion')([x_csv, x_img, x_wav])
    
    # Subimos ligeramente la fusión inicial para acomodar los nuevos datos de la imagen
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(fusion)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(N_CLASES, activation='softmax', name='output')(x)

    modelo = Model(inputs=[inp_csv, inp_img, inp_wav], outputs=salida, name='clasificador_hibrido_potenciado')

    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo, base_cnn
    """def construir_modelo_hibrido():
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import regularizers

    print("\n" + "=" * 60)
    print("  CONSTRUYENDO MODELO HÍBRIDO MEJORADO")
    print("=" * 60)

    # ── Rama A: CSV features ──────────────────────────────────────
    inp_csv = layers.Input(shape=(N_FEATURES,), name='input_csv')
    x_csv = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005))(inp_csv) 
    x_csv = layers.BatchNormalization()(x_csv)
    x_csv = layers.Dropout(0.4)(x_csv)
    x_csv = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x_csv)
    x_csv = layers.BatchNormalization()(x_csv)

    # ── Rama B: Imagen espectrograma ──────────────────────────────
"""
    """
    inp_img = layers.Input(shape=(*IMG_SIZE, 3), name='input_img')
    base_cnn = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_cnn.trainable = False 
    x_img = base_cnn(inp_img, training=False)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dropout(0.5)(x_img) # Dropout más alto para forzar generalización
    """    
    """   
    # Nueva Rama B
    inp_img = layers.Input(shape=(*IMG_SIZE, 3), name='input_img')
    
    # NUEVO: Data Augmentation nativo de Keras
    x_img_aug = layers.RandomFlip("horizontal")(inp_img)
    # Movemos ligeramente el espectrograma en el eje del tiempo (ancho)
    x_img_aug = layers.RandomTranslation(height_factor=0.0, width_factor=0.1)(x_img_aug)

    base_cnn = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_cnn.trainable = False 
    
    # IMPORTANTE: Pasamos x_img_aug en lugar de inp_img
    x_img = base_cnn(x_img_aug, training=False) 
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dropout(0.5)(x_img)
    # ── Rama C: WAV features ──────────────────────────────────────
    inp_wav = layers.Input(shape=(N_FEATURES,), name='input_wav')
    x_wav = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.005))(inp_wav)
    x_wav = layers.BatchNormalization()(x_wav)
    x_wav = layers.Dropout(0.4)(x_wav)
    x_wav = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005))(x_wav)
    x_wav = layers.BatchNormalization()(x_wav)

    # ── Fusión de las 3 ramas ─────────────────────────────────────
    fusion = layers.Concatenate(name='fusion')([x_csv, x_img, x_wav])
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.002))(fusion)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.002))(x)
    x = layers.Dropout(0.4)(x)
    salida = layers.Dense(N_CLASES, activation='softmax', name='output')(x)

    modelo = Model(inputs=[inp_csv, inp_img, inp_wav], outputs=salida, name='clasificador_hibrido_v2')

    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo, base_cnn
    """
# ══════════════════════════════════════════════════════════════════
# PARTE 4: ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════
"""
def entrenar(guardar_en: str = MODELO_PATH):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Cargar datos
    X_csv, X_img, X_wav, y = cargar_dataset_completo()

    # Split estratificado
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    # Escalar las ramas numéricas (CSV y WAV)
    scaler_csv = StandardScaler()
    scaler_wav = StandardScaler()

    X_csv_tr = scaler_csv.fit_transform(X_csv[idx_tr])
    X_csv_te = scaler_csv.transform(X_csv[idx_te])

    X_wav_tr = scaler_wav.fit_transform(X_wav[idx_tr])
    X_wav_te = scaler_wav.transform(X_wav[idx_te])

    X_img_tr = X_img[idx_tr]
    X_img_te = X_img[idx_te]

    y_tr = y[idx_tr]
    y_te = y[idx_te]

    print(f"\n  Train: {len(idx_tr)}  |  Test: {len(idx_te)}")

    # Guardar scalers
    joblib.dump({'csv': scaler_csv, 'wav': scaler_wav}, SCALER_PATH)

    # Construir modelo
    modelo, base_cnn = construir_modelo_hibrido()

    inputs_tr = {'input_csv': X_csv_tr, 'input_img': X_img_tr, 'input_wav': X_wav_tr}
    inputs_te = {'input_csv': X_csv_te, 'input_img': X_img_te, 'input_wav': X_wav_te}

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    # ── FASE 1: Solo cabeza ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FASE 1: Entrenando cabeza (CNN congelada)")
    print("=" * 60)

    h1 = modelo.fit(
        inputs_tr, y_tr,
        epochs=EPOCHS_FASE1,
        batch_size=BATCH_SIZE,
        validation_data=(inputs_te, y_te),
        callbacks=callbacks,
        verbose=1
    )

    # ── FASE 2: Fine-tuning últimas capas CNN ─────────────────────
    print("\n" + "=" * 60)
    print("  FASE 2: Fine-tuning últimas 15 capas de MobileNetV2")
    print("=" * 60)

    base_cnn.trainable = True
    for layer in base_cnn.layers[:-15]: # CAMBIADO a -15
        layer.trainable = False

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-6), # Tasa ultra baja para no destruir pesos
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    h2 = modelo.fit(
        inputs_tr, y_tr,
        epochs=EPOCHS_FASE2,
        batch_size=BATCH_SIZE,
        validation_data=(inputs_te, y_te),
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=10,
                                  restore_best_weights=True, verbose=1),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                     patience=4, verbose=1)],
        verbose=1
    )

    # Evaluación final
    print("\n" + "=" * 60)
    print("  EVALUACIÓN FINAL")
    print("=" * 60)
    loss, acc = modelo.evaluate(inputs_te, y_te, verbose=0)
    print(f"\n  ✅ Accuracy en test: {acc:.4f}  ({acc*100:.1f}%)")

    # Guardar modelo y clases
    modelo.save(guardar_en)
    with open(CLASES_PATH, 'w') as f:
        json.dump({str(i): g for i, g in enumerate(GENEROS)}, f)

    print(f"  💾 Modelo guardado: {guardar_en}")
    print(f"  💾 Scalers guardados: {SCALER_PATH}")
    print(f"  💾 Clases guardadas: {CLASES_PATH}")

    # Gráfica
    _grafica_entrenamiento(h1, h2)
    return modelo
"""

# ══════════════════════════════════════════════════════════════════
# PARTE 4: ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════

def entrenar(guardar_en: str = MODELO_PATH):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Cargar datos
    X_csv, X_img, X_wav, y = cargar_dataset_completo()

    # Split estratificado
    idx = np.arange(len(y))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)

    # Escalar las ramas numéricas (CSV y WAV)
    scaler_csv = StandardScaler()
    scaler_wav = StandardScaler()

    X_csv_tr = scaler_csv.fit_transform(X_csv[idx_tr])
    X_csv_te = scaler_csv.transform(X_csv[idx_te])

    X_wav_tr = scaler_wav.fit_transform(X_wav[idx_tr])
    X_wav_te = scaler_wav.transform(X_wav[idx_te])

    X_img_tr = X_img[idx_tr]
    X_img_te = X_img[idx_te]

    y_tr = y[idx_tr]
    y_te = y[idx_te]

    print(f"\n  Train: {len(idx_tr)}  |  Test: {len(idx_te)}")

    # Guardar scalers
    joblib.dump({'csv': scaler_csv, 'wav': scaler_wav}, SCALER_PATH)

    # Construir modelo
    modelo, base_cnn = construir_modelo_hibrido()

    inputs_tr = {'input_csv': X_csv_tr, 'input_img': X_img_tr, 'input_wav': X_wav_tr}
    inputs_te = {'input_csv': X_csv_te, 'input_img': X_img_te, 'input_wav': X_wav_te}

    # Le damos más "paciencia" para que encuentre el pico de precisión sin rendirse
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    # ── ENTRENAMIENTO ÚNICO ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ENTRENANDO MODELO (CNN CONGELADA COMO EXTRACTOR PERFECTO)")
    print("=" * 60)

    # Dejamos que llegue hasta 50 épocas si lo necesita
    h1 = modelo.fit(
        inputs_tr, y_tr,
        epochs=50,
        batch_size=BATCH_SIZE,
        validation_data=(inputs_te, y_te),
        callbacks=callbacks,
        verbose=1
    )

    # Evaluación final
    print("\n" + "=" * 60)
    print("  EVALUACIÓN FINAL")
    print("=" * 60)
    loss, acc = modelo.evaluate(inputs_te, y_te, verbose=0)
    print(f"\n  ✅ Accuracy en test: {acc:.4f}  ({acc*100:.1f}%)")

    # Guardar modelo y clases
    modelo.save(guardar_en)
    with open(CLASES_PATH, 'w') as f:
        json.dump({str(i): g for i, g in enumerate(GENEROS)}, f)

    print(f"  💾 Modelo guardado: {guardar_en}")
    print(f"  💾 Scalers guardados: {SCALER_PATH}")
    print(f"  💾 Clases guardadas: {CLASES_PATH}")

    # Gráfica
    _grafica_entrenamiento(h1)
    return modelo


def _grafica_entrenamiento(h1):
    acc    = h1.history['accuracy']
    val    = h1.history['val_accuracy']

    plt.figure(figsize=(10, 5))
    plt.plot(acc, label='Train')
    plt.plot(val, label='Validación')
    plt.title('Accuracy — Modelo Híbrido (Óptimo)')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('entrenamiento_hibrido.png', dpi=150)
    print("  📊 Gráfica: entrenamiento_hibrido.png")

# ══════════════════════════════════════════════════════════════════
# PARTE 5: PREDICCIÓN CON VENTANAS DESLIZANTES
# ══════════════════════════════════════════════════════════════════

def predecir_cancion(ruta_audio: str,
                     ventana_seg: int = 30,
                     salto_seg: int = 15) -> dict:
    """
    Predice el género de cualquier .wav o .mp3 usando los 3 tipos de datos.

    Para canciones largas usa ventanas deslizantes y promedia resultados.

    Uso:
        resultado = predecir_cancion("mi_cancion.wav")
        print(resultado['genero'], resultado['confianza'])
    """
    import tensorflow as tf
    import librosa

    if not os.path.exists(ruta_audio):
        raise FileNotFoundError(f"No se encuentra: {ruta_audio}")

    # Entrenar si no existe modelo
    if not os.path.exists(MODELO_PATH):
        print("Modelo no encontrado. Entrenando primero...")
        entrenar()

    # Cargar modelo, scalers y clases
    modelo   = tf.keras.models.load_model(MODELO_PATH)
    scalers  = joblib.load(SCALER_PATH)
    with open(CLASES_PATH) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    # Cargar audio completo
    print(f"\n  Cargando: {os.path.basename(ruta_audio)}...")
    y_completo, sr = librosa.load(ruta_audio, mono=True)
    duracion = len(y_completo) / sr
    print(f"  Duración: {duracion:.1f}s ({duracion/60:.1f} min)")

    # Dividir en ventanas deslizantes
    ventana_s = int(ventana_seg * sr)
    salto_s   = int(salto_seg * sr)
    inicios   = list(range(0, max(1, len(y_completo) - ventana_s + 1), salto_s))
    fragmentos = [y_completo[i: i + ventana_s] for i in inicios] or [y_completo]

    print(f"  Analizando {len(fragmentos)} fragmentos de {ventana_seg}s...\n")

    probas_acum = np.zeros(N_CLASES)

    for i, frag in enumerate(fragmentos):
        t_ini = inicios[i] // sr if i < len(inicios) else 0
        t_fin = min(t_ini + ventana_seg, int(duracion))
        print(f"  [{i+1:2d}/{len(fragmentos)}] "
              f"{t_ini//60:02d}:{t_ini%60:02d} → {t_fin//60:02d}:{t_fin%60:02d}",
              end="  ")

        # Extraer features del fragmento (ramas CSV y WAV usan la misma función)
        feats = extraer_features_wav(frag, sr).reshape(1, -1)
        feats_csv_s = scalers['csv'].transform(feats)
        feats_wav_s = scalers['wav'].transform(feats)

        # Generar espectrograma (rama imagen)
        img = audio_a_espectrograma(frag, sr)[np.newaxis]  # (1, H, W, 3)

        # Predecir
        probas = modelo.predict(
            {'input_csv': feats_csv_s,
             'input_img': img,
             'input_wav': feats_wav_s},
            verbose=0
        )[0]

        idx_max = np.argmax(probas)
        print(f"→ {idx_to_class[idx_max]:<12} ({probas[idx_max]*100:.1f}%)")
        probas_acum += probas

    # Resultado final por promedio de probabilidades
    probas_medias = probas_acum / len(fragmentos)
    ranking = sorted(enumerate(probas_medias), key=lambda x: x[1], reverse=True)
    top3    = [(idx_to_class[i], round(float(p) * 100, 1)) for i, p in ranking[:3]]

    resultado = {
        'genero':      top3[0][0],
        'confianza':   top3[0][1],
        'top3':        top3,
        'todas':       {idx_to_class[i]: round(float(p)*100,1) for i, p in ranking},
        'fragmentos':  len(fragmentos),
        'duracion':    duracion,
    }

    print("\n" + "═" * 50)
    print(f"  🎵  Género detectado : {resultado['genero'].upper()}")
    print(f"  📊  Confianza media  : {resultado['confianza']}%")
    print(f"  🔢  Fragmentos       : {resultado['fragmentos']}")
    print(f"\n  Top 3:")
    for i, (g, p) in enumerate(top3):
        barra = "█" * int(p / 4)
        print(f"    {i+1}. {g:<12} {barra} {p}%")
    print("═" * 50 + "\n")

    return resultado


# ══════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if len(sys.argv) > 1:
        # ── Modo predicción ──────────────────────────────────────
        predecir_cancion(sys.argv[1])

    else:
        # ── Modo entrenamiento ───────────────────────────────────
        print("CLASIFICADOR HÍBRIDO DE GÉNEROS MUSICALES")
        print("=" * 60)

        faltan = []
        if not os.path.exists(CSV_PATH):
            faltan.append(f"  ✗ CSV:     '{CSV_PATH}' no encontrado")
        if not os.path.exists(IMAGES_DIR):
            faltan.append(f"  ✗ Imágenes: '{IMAGES_DIR}/' no encontrada → descomprime images_original.zip")
        if not os.path.exists(WAVS_DIR):
            faltan.append(f"  ✗ WAVs:    '{WAVS_DIR}/' no encontrada → descomprime los 3 zips aquí")

        if faltan:
            print("\nFaltan archivos necesarios:")
            for m in faltan: print(m)
            print("\nEstructura esperada:")
            print("  features_3_sec.csv")
            print("  images_original/blues/ classical/ ... rock/")
            print("  genres_original/blues/ classical/ ... rock/")
            sys.exit(1)

        if os.path.exists(MODELO_PATH):
            print(f"\n✅ Modelo ya entrenado en '{MODELO_PATH}'")
            print("   Bórralo y vuelve a ejecutar si quieres reentrenar.\n")
            print("Para predecir:")
            print("  python clasificador_hibrido.py mi_cancion.wav")
        else:
            print("\n⚠️  El entrenamiento puede tardar 10-20 min sin GPU.\n")
            entrenar()
            print("\n✅ Listo. Para predecir:")
            print("  python clasificador_hibrido.py mi_cancion.wav")
