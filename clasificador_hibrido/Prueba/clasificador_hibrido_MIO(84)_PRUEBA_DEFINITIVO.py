"""
╔══════════════════════════════════════════════════════════════════════╗
║         CLASIFICADOR HÍBRIDO MUSICAL (VERSIÓN OPTIMIZADA)            ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import regularizers
import tensorflow as tf

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ─────────────────────────────────────────────────────────────────
CSV_PATH      = "features_30_sec.csv"   # <-- Usamos el de 30s (más rápido)
IMAGES_DIR    = "images_original"
WAVS_DIR      = "genres_original"
MODELO_PATH   = "modelo_hibrido_definitivo.keras"
SCALER_PATH   = "scaler_hibrido_definitivo.pkl"
CLASES_PATH   = "clases_hibrido_definitivo.json"

IMG_SIZE      = (128, 128)   # <-- Reducido para máxima velocidad
BATCH_SIZE    = 32         # <-- Mayor batch para usar mejor la CPU/GPU
EPOCHS_FASE1  = 30         # Entrenamiento cabeza
EPOCHS_FASE2  = 15         # Fine-tuning (El secreto para el 90%)
N_CLASES      = 10
N_FEATURES    = 57
GENEROS       = sorted(['blues','classical','country','disco','hiphop',
                         'jazz','metal','pop','reggae','rock'])

# ══════════════════════════════════════════════════════════════════
# 1. EXTRACCIÓN (Igual que antes)
# ══════════════════════════════════════════════════════════════════
def extraer_features_wav(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa
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
    import librosa.display
    from PIL import Image
    import io
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

# ══════════════════════════════════════════════════════════════════
# 2. CARGA DE DATASET (Optimizado para 30s)
# ══════════════════════════════════════════════════════════════════
def cargar_dataset_completo():
    import librosa
    from PIL import Image

    print("\n  [1/3] Cargando CSV (30 segundos)...")
    df = pd.read_csv(CSV_PATH)
    feat_cols = [c for c in df.columns if c not in ['filename','length','label']]
    
    # "blues.00000.wav" -> "blues.00000"
    df['cancion_base'] = df['filename'].apply(lambda x: str(x).split('.wav')[0])

    X_csv_list, X_img_list, X_wav_list, y_list = [], [], [], []
    genero_a_idx = {g: i for i, g in enumerate(GENEROS)}
    errores = 0

    print("  [2/3] Procesando Audios e Imágenes (puede tardar un poco)...")
    for _, fila in df.iterrows():
        genero = fila['label']
        cancion_id = fila['cancion_base']
        numero = cancion_id.split('.')[1]

        ruta_img = os.path.join(IMAGES_DIR, genero, f"{genero}{numero}.png")
        ruta_wav = os.path.join(WAVS_DIR, genero, f"{cancion_id}.wav")

        if not os.path.exists(ruta_img) or not os.path.exists(ruta_wav):
            errores += 1
            continue

        try:
            # Cargar imagen
            img = Image.open(ruta_img).convert('RGB').resize(IMG_SIZE)
            img_arr = preprocess_input(np.array(img, dtype=np.float32))
            
            # Cargar audio
            y_audio, sr = librosa.load(ruta_wav, mono=True, duration=30)
            wav_feats = extraer_features_wav(y_audio, sr)
            
            X_csv_list.append(fila[feat_cols].values.astype(np.float32))
            X_img_list.append(img_arr)
            X_wav_list.append(wav_feats)
            y_list.append(genero_a_idx[genero])
        except Exception:
            errores += 1

    print(f"     Cargadas: {len(X_csv_list)} | Errores: {errores}")
    return np.stack(X_csv_list), np.stack(X_img_list), np.stack(X_wav_list), np.array(y_list)

# ══════════════════════════════════════════════════════════════════
# 3. MODELO Y ENTRENAMIENTO (RESTAUADO A MÁXIMA PRECISIÓN)
# ══════════════════════════════════════════════════════════════════
def construir_modelo():
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras import regularizers

    # ── Rama A: CSV features
    inp_csv = layers.Input(shape=(N_FEATURES,))
    x_csv = layers.GaussianNoise(0.01)(inp_csv)
    x_csv = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_csv)
    x_csv = layers.BatchNormalization()(x_csv)
    x_csv = layers.Dropout(0.4)(x_csv)
    x_csv = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_csv)
    x_csv = layers.BatchNormalization()(x_csv)

    # ── Rama B: Imagen espectrograma
    inp_img = layers.Input(shape=(128, 128, 3))
    x_img_aug = layers.RandomTranslation(height_factor=0.0, width_factor=0.05)(inp_img)
    
    base_cnn = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
    base_cnn.trainable = False  
    
    x_img = base_cnn(x_img_aug, training=False)
    x_img = layers.GlobalAveragePooling2D()(x_img)
    x_img = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_img)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dropout(0.5)(x_img)

    # ── Rama C: WAV features
    inp_wav = layers.Input(shape=(N_FEATURES,))
    x_wav = layers.GaussianNoise(0.01)(inp_wav)
    x_wav = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_wav)
    x_wav = layers.BatchNormalization()(x_wav)
    x_wav = layers.Dropout(0.4)(x_wav)
    x_wav = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x_wav)
    x_wav = layers.BatchNormalization()(x_wav)

    # ── Fusión
    fusion = layers.Concatenate()([x_csv, x_img, x_wav])
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(fusion)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(N_CLASES, activation='softmax')(x)

    modelo = Model(inputs=[inp_csv, inp_img, inp_wav], outputs=salida)
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return modelo, base_cnn

def entrenar():
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    X_csv, X_img, X_wav, y = cargar_dataset_completo()
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)

    scaler_csv, scaler_wav = StandardScaler(), StandardScaler()
    X_csv_tr = scaler_csv.fit_transform(X_csv[idx_tr])
    X_csv_te = scaler_csv.transform(X_csv[idx_te])
    X_wav_tr = scaler_wav.fit_transform(X_wav[idx_tr])
    X_wav_te = scaler_wav.transform(X_wav[idx_te])

    joblib.dump({'csv': scaler_csv, 'wav': scaler_wav}, SCALER_PATH)

    modelo, base_cnn = construir_modelo()
    
    # ¡AQUÍ ESTÁ LA MAGIA DEL ARREGLO! (Listas en lugar de diccionarios)
    inputs_tr = [X_csv_tr, X_img[idx_tr], X_wav_tr]
    inputs_te = [X_csv_te, X_img[idx_te], X_wav_te]

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
    ]

    print("\n" + "="*60)
    print("  ENTRENANDO MODELO (MODO ALTA PRECISIÓN)")
    print("="*60)

    h1 = modelo.fit(
        inputs_tr, y[idx_tr], 
        epochs=50, 
        batch_size=32,
        validation_data=(inputs_te, y[idx_te]),
        callbacks=callbacks
    )

    loss, acc = modelo.evaluate(inputs_te, y[idx_te], verbose=0)
    print(f"\n  ✅ Accuracy Final en test: {acc*100:.1f}%")

    modelo.save(MODELO_PATH)
    with open(CLASES_PATH, 'w') as f:
        json.dump({str(i): g for i, g in enumerate(GENEROS)}, f)

# ══════════════════════════════════════════════════════════════════
# 4. PREDICCIÓN CON VENTANAS (10 SEGUNDOS)
# ══════════════════════════════════════════════════════════════════
def predecir(ruta_audio: str):
    import librosa
    modelo = tf.keras.models.load_model(MODELO_PATH)
    scalers = joblib.load(SCALER_PATH)
    with open(CLASES_PATH) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    print(f"\n  Analizando: {os.path.basename(ruta_audio)}...")
    y_full, sr = librosa.load(ruta_audio, mono=True)
    
    # Ventanas de 10 segundos
    ven_s = 20 * sr #antes 10
    salto_s = 10 * sr #Antes5
    inicios = list(range(0, max(1, len(y_full) - ven_s), salto_s))
    
    probas_acum = np.zeros(N_CLASES)
    
    for i, t_ini in enumerate(inicios):
        frag = y_full[t_ini : t_ini + ven_s]
        if len(frag) < ven_s: break # Ignorar colas cortas
        
        feats = extraer_features_wav(frag, sr).reshape(1, -1)
        csv_s = scalers['csv'].transform(feats)
        wav_s = scalers['wav'].transform(feats)
        img = audio_a_espectrograma(frag, sr)[np.newaxis]
        
        probas = modelo.predict([csv_s, img, wav_s], verbose=0)[0]
        probas_acum += probas
        print(f"  [{i+1}/{len(inicios)}] -> {idx_to_class[np.argmax(probas)]}")

    ranking = sorted(enumerate(probas_acum / len(inicios)), key=lambda x: x[1], reverse=True)
    print("\n" + "═" * 50)
    print(f"  🎵  GÉNERO: {idx_to_class[ranking[0][0]].upper()} ({ranking[0][1]*100:.1f}%)")
    print(f"  🎵  SUB : {idx_to_class[ranking[1][0]].upper()} ({ranking[1][1]*100:.1f}%)")
    print("═" * 50)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predecir(sys.argv[1])
    else:
        if not os.path.exists(MODELO_PATH):
            entrenar()
        else:
            print("Modelo ya entrenado. Para predecir usa: python clasificador.py audio.mp3")
