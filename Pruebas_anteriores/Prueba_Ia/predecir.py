import librosa
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# 1. Copiamos la MISMA función de extracción que usaste en el entrenamiento
def extract_features_song(ruta_cancion):
    try:
        audio_data, sampling_rate = librosa.load(ruta_cancion)
        
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_var = np.var(mfccs, axis=1)
        
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sampling_rate)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_var = np.var(chroma, axis=1)
        
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate)
        centroid_mean = np.mean(centroid, axis=1)
        centroid_var = np.var(centroid, axis=1)
        
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        zcr_mean = np.mean(zcr, axis=1)
        zcr_var = np.var(zcr, axis=1)
        
        features = np.hstack((mfccs_mean, mfccs_var, chroma_mean, chroma_var, centroid_mean, centroid_var, zcr_mean, zcr_var))
        return features
    except Exception as e:
        print(f"Error al procesar el audio: {e}")
        return None

# 2. Función principal para predecir
def predecir_genero(ruta_archivo):
    print(f"Analizando: {ruta_archivo}...")
    
    # A. Cargar el "cerebro" guardado
    modelo = load_model('mi_modelo_musical.h5')
    scaler = joblib.load('mi_escalador.pkl')
    etiquetas = np.load('mis_etiquetas.npy', allow_pickle=True)
    
    # B. Extraer las características matemáticas de la canción nueva
    caracteristicas = extract_features_song(ruta_archivo)
    
    if caracteristicas is not None:
        # C. Darle el formato correcto (1 fila, X columnas) y escalarlo
        caracteristicas = caracteristicas.reshape(1, -1)
        caracteristicas_escaladas = scaler.transform(caracteristicas)
        
        # D. Hacer la predicción
        prediccion_array = modelo.predict(caracteristicas_escaladas)
        indice_ganador = np.argmax(prediccion_array)
        probabilidad = np.max(prediccion_array) * 100
        
        # E. Mostrar el resultado
        genero_ganador = etiquetas[indice_ganador]
        print(f"\n🎵 ¡Predicción completada!")
        print(f"El género es: **{genero_ganador.upper()}** (Seguridad: {probabilidad:.2f}%)")

# =====================================================================
# Pruébalo aquí con cualquier canción que tengas en tu PC
# =====================================================================
cancion_de_prueba = "../Musica_prueba/ytmp3free.cc_earth-wind-fire-boogie-wonderland-official-video-youtubemp3free.org.mp3" 
predecir_genero(cancion_de_prueba)