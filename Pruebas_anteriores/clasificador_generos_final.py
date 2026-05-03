"""
╔══════════════════════════════════════════════════════════════════╗
║         CLASIFICADOR DE GÉNEROS MUSICALES — VERSIÓN FINAL        ║
║   Entrena con los CSVs  +  predice cualquier archivo .wav/.mp3   ║
╚══════════════════════════════════════════════════════════════════╝

REQUISITOS:
    pip install librosa scikit-learn pandas numpy joblib seaborn

USO RÁPIDO:
    1. Ejecuta este script completo → entrena y guarda el modelo
    2. Para predecir una canción tuya:
           resultado = predecir_cancion("mi_cancion.wav")
           print(resultado)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import librosa

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estas rutas si es necesario
# ─────────────────────────────────────────────────────────────────
CSV_PATH       = "./P05/Data/features_3_sec.csv"   # El CSV de 3 segundos (9990 filas)
MODELO_PATH    = "./P05/Data/modelo_generos.pkl"   # Donde se guarda el modelo entrenado


# ══════════════════════════════════════════════════════════════════
# PARTE 1: EXTRACCIÓN DE FEATURES
#   Esta función convierte cualquier .wav/.mp3 en el mismo vector
#   de 57 features que tiene el CSV. Es el puente entre un audio
#   real y el modelo entrenado.
# ══════════════════════════════════════════════════════════════════

def extraer_features_de_audio(ruta_audio: str) -> np.ndarray:
    """
    Carga un archivo de audio y extrae las mismas 57 features
    que contiene el CSV de GTZAN.

    Parámetros:
        ruta_audio: ruta al archivo .wav o .mp3

    Devuelve:
        Array numpy de shape (57,) listo para pasarle al modelo
    """
    y, sr = librosa.load(ruta_audio, duration=30)  # Máximo 30 segundos

    features = []

    # 1. Chroma STFT (armonía — las 12 notas de la escala cromática)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features += [np.mean(chroma), np.var(chroma)]

    # 2. RMS Energy (volumen / energía de la señal)
    rms = librosa.feature.rms(y=y)
    features += [np.mean(rms), np.var(rms)]

    # 3. Spectral Centroid (brillo del sonido — "centro de masa")
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features += [np.mean(centroid), np.var(centroid)]

    # 4. Spectral Bandwidth (anchura del espectro)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features += [np.mean(bandwidth), np.var(bandwidth)]

    # 5. Spectral Rolloff (frecuencia donde se concentra el 85% de la energía)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features += [np.mean(rolloff), np.var(rolloff)]

    # 6. Zero Crossing Rate (ritmo/percusión — cuántas veces cruza el cero)
    zcr = librosa.feature.zero_crossing_rate(y)
    features += [np.mean(zcr), np.var(zcr)]

    # 7. Harmony y Perceptr (componentes armónicas y de percusión)
    harmony, perceptr = librosa.effects.hpss(y)
    features += [np.mean(harmony), np.var(harmony)]
    features += [np.mean(perceptr), np.var(perceptr)]

    # 8. Tempo (BPM estimado)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features += [float(tempo)]

    # 9. MFCCs 1–20 (media y varianza de cada coeficiente)
    #    Representan el timbre y la textura espectral del audio
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(20):
        features += [np.mean(mfccs[i]), np.var(mfccs[i])]

    # Total: 2+2+2+2+2+2+2+2+1+(20×2) = 57 features ✓
    return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════
# PARTE 2: ENTRENAMIENTO DEL MODELO
# ══════════════════════════════════════════════════════════════════

def entrenar_modelo(csv_path: str = CSV_PATH, guardar_en: str = MODELO_PATH):
    """
    Entrena el clasificador SVM con los datos del CSV y lo guarda en disco.
    Solo necesitas ejecutar esto una vez.
    """
    print("=" * 55)
    print("  ENTRENANDO MODELO")
    print("=" * 55)

    # Cargar CSV
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ['filename', 'length', 'label']]
    X = df[feature_cols].values

    le = LabelEncoder()
    y = le.fit_transform(df['label'])

    print(f"  Muestras: {len(X)}  |  Features: {len(feature_cols)}")
    print(f"  Géneros:  {list(le.classes_)}\n")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Escalar y entrenar SVM
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("  Entrenando SVM (puede tardar ~30 segundos)...")
    modelo = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    modelo.fit(X_train_s, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  ✅ Accuracy en test: {acc:.4f}  ({acc*100:.1f}%)\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Guardar todo lo necesario para predecir después
    paquete = {
        'modelo':         modelo,
        'scaler':         scaler,
        'label_encoder':  le,
        'feature_cols':   feature_cols,
    }
    joblib.dump(paquete, guardar_en)
    print(f"  💾 Modelo guardado en: {guardar_en}")

    return paquete


# ══════════════════════════════════════════════════════════════════
# PARTE 3: PREDICCIÓN DE UNA CANCIÓN
# ══════════════════════════════════════════════════════════════════

def predecir_cancion(ruta_audio: str, modelo_path: str = MODELO_PATH) -> dict:
    """
    Predice el género musical de cualquier archivo .wav o .mp3.

    Parámetros:
        ruta_audio:   ruta al archivo de audio (ej: "mi_cancion.wav")
        modelo_path:  ruta al modelo guardado (por defecto "modelo_generos.pkl")

    Devuelve un diccionario con:
        - 'genero':       el género más probable
        - 'confianza':    porcentaje de confianza (0-100)
        - 'top3':         los 3 géneros más probables con sus porcentajes

    Ejemplo de uso:
        resultado = predecir_cancion("mi_cancion.wav")
        print(f"Género: {resultado['genero']}  ({resultado['confianza']:.1f}% de confianza)")
        print(f"Top 3: {resultado['top3']}")
    """
    # Verificar que existe el archivo
    if not os.path.exists(ruta_audio):
        raise FileNotFoundError(f"No se encuentra el archivo: {ruta_audio}")

    # Cargar modelo si existe, entrenar si no
    if not os.path.exists(modelo_path):
        print(f"Modelo no encontrado en '{modelo_path}'. Entrenando primero...")
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(
                f"Tampoco se encuentra el CSV '{CSV_PATH}'. "
                "Asegúrate de tener 'features_3_sec.csv' en la misma carpeta."
            )
        entrenar_modelo()

    paquete = joblib.load(modelo_path)
    modelo  = paquete['modelo']
    scaler  = paquete['scaler']
    le      = paquete['label_encoder']

    # Extraer features del audio
    print(f"  Analizando: {os.path.basename(ruta_audio)}...")
    features = extraer_features_de_audio(ruta_audio).reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predicción con probabilidades por género
    probas   = modelo.predict_proba(features_scaled)[0]
    generos  = le.classes_

    # Ordenar por probabilidad descendente
    ranking  = sorted(zip(generos, probas), key=lambda x: x[1], reverse=True)
    top3     = [(g, round(p * 100, 1)) for g, p in ranking[:3]]

    resultado = {
        'genero':    ranking[0][0],
        'confianza': round(ranking[0][1] * 100, 1),
        'top3':      top3,
        'todas':     {g: round(p * 100, 1) for g, p in ranking},
    }

    # Mostrar resultado de forma visual
    print("\n" + "─" * 40)
    print(f"  🎵  Género detectado: {resultado['genero'].upper()}")
    print(f"  📊  Confianza:        {resultado['confianza']}%")
    print(f"\n  Top 3 más probables:")
    for i, (g, p) in enumerate(top3):
        barra = "█" * int(p / 5)
        print(f"    {i+1}. {g:<12} {barra} {p}%")
    print("─" * 40 + "\n")

    return resultado


# ══════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── PASO 1: Entrenar y guardar el modelo (solo la primera vez) ──
    if not os.path.exists(MODELO_PATH):
        entrenar_modelo()
    else:
        print(f"✅ Modelo ya entrenado encontrado en '{MODELO_PATH}'")
        print("   (Borra el archivo y vuelve a ejecutar si quieres reentrenar)\n")

    # ── PASO 2: Predecir una canción ──
    #
    # Opción A — pasarle la ruta directamente en el código:
    #
    #   resultado = predecir_cancion("mi_cancion.wav")
    #
    # Opción B — pasarla como argumento al ejecutar el script:
    #
    #   python clasificador_generos_final.py mi_cancion.wav
    #

    if len(sys.argv) > 1:
        # Llamada desde terminal: python clasificador_generos_final.py cancion.wav
        ruta = sys.argv[1]
        predecir_cancion(ruta)
    else:
        print("─" * 55)
        print("  CÓMO USAR PARA PREDECIR UNA CANCIÓN:")
        print("─" * 55)
        print()
        print("  Opción 1 — desde terminal:")
        print("    python clasificador_generos_final.py mi_cancion.wav")
        print()
        print("  Opción 2 — desde tu código Python:")
        print("    from clasificador_generos_final import predecir_cancion")
        print("    resultado = predecir_cancion('mi_cancion.wav')")
        print("    print(resultado['genero'], resultado['confianza'])")
        print()
