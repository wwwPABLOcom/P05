
########################################################## PARTE 1: Carga de los datos ############################################################
import librosa
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt

# Importaciones de Keras para la Red Neuronal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical

def display_mfcc(file_path, title):
    # 1. Cargamos el archivo de audio
    # y = la señal de audio, sr = sample rate (frecuencia de muestreo)
    y, sr = librosa.load(file_path)
    
    # 2. Extraemos los MFCCs (por defecto suele extraer 20)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # 3. Visualización con specshow
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'MFCC Spectrogram: {title}')
    plt.tight_layout()
    plt.show()
    
    return mfccs

########################################################### PARTE 2 Preprocesado de los datos ####################################################
import librosa
import numpy as np
def extract_features_song(ruta_cancion):
    try:
        audio_data, sampling_rate = librosa.load(ruta_cancion)
        
        # 1. MFCCs (Timbre) - 40 características
        # Representa la forma general en la que suena el audio (espectro vocal/instrumental)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs, axis=1)
        
        # 2. Chroma Frequencies (Armonía) - 12 características
        # Representa las 12 notas musicales (Do, Do#, Re...) de la escala cromática
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sampling_rate)
        chroma_mean = np.mean(chroma, axis=1)
        
        # 3. Spectral Centroid (Brillo) - 1 característica
        # Indica dónde está el "centro de masa" del sonido (los tonos agudos tiran hacia arriba)
        centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate)
        centroid_mean = np.mean(centroid, axis=1)
        
        # 4. Zero Crossing Rate (Ritmo/Percusión/Ruido) - 1 característica
        # Mide cuántas veces la señal de audio pasa por el cero (muy útil para el Metal o el Hiphop)
        zcr = librosa.feature.zero_crossing_rate(y=audio_data)
        zcr_mean = np.mean(zcr, axis=1)
        
        # Apilamos todas las medias en un único vector unidimensional
        # Total: 40 + 12 + 1 + 1 = 54 características por canción
        features = np.hstack((mfccs_mean, chroma_mean, centroid_mean, zcr_mean))
        
        return features
        
    except Exception as e:
        print(f"Error al procesar {ruta_cancion}: {e}")
        return None
"""
def extract_features_song(ruta_cancion):
    try:
        audio_data, sampling_rate = librosa.load(ruta_cancion)
        mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc_features, axis=1)
        
        # Devuelve la media directamente, SIN normalizar aquí
        return mfcc_mean 
        
    except Exception as e:
        print(f"Error al procesar {ruta_cancion}: {e}")
        return None
"""
"""
def extract_features_song(ruta_cancion):
    try:
        audio_data, sampling_rate = librosa.load(ruta_cancion)
        # Utilizando un número fijo de coeficientes MFCC (por ejemplo, 40) basado en la práctica común.
        mfcc_features = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=40)
        # Rellenar o truncar a un tamaño fijo (25000 como se menciona en el texto).
        longitud_maxima_mfcc = 1300
        if mfcc_features.shape[1] < longitud_maxima_mfcc:
            ancho_relleno = longitud_maxima_mfcc - mfcc_features.shape[1]
            mfcc_features = np.pad(mfcc_features, pad_width=((0, 0), (0, ancho_relleno)), mode='constant')
        else:
            mfcc_features = mfcc_features[:, :longitud_maxima_mfcc]

        # Normalizar los coeficientes MFCC para que estén entre -1 y 1.
        mfcc_normalizados = 2 * ((mfcc_features - mfcc_features.min()) / (mfcc_features.max() - mfcc_features.min())) - 1
        return mfcc_normalizados
    except Exception as e:
        print(f"Error al procesar {ruta_cancion}: {e}")
        return None
"""
import glob
import os
import numpy as np
from keras.utils import to_categorical

def generate_features_and_labels(ruta_datos='Data/genres_original'):
    todas_las_caracteristicas = []
    todas_las_etiquetas = []
    generos = sorted(os.listdir(ruta_datos))

    # Iterar a través de cada género musical
    for i, genero in enumerate(generos):
        ruta_genero = os.path.join(ruta_datos, genero)
        # Saltar si no es un directorio (ej., .DS_Store)
        if not os.path.isdir(ruta_genero):
            continue

        print(f"Procesando género: {genero}")
        # Buscar todos los archivos .wav en la carpeta del género
        for archivo_cancion in glob.glob(os.path.join(ruta_genero, '*.wav')):
            # Extraer características MFCC de la canción
            mfcc_extraidos = extract_features_song(archivo_cancion)
            if mfcc_extraidos is not None:
                todas_las_caracteristicas.append(mfcc_extraidos)
                todas_las_etiquetas.append(genero)

    # Convertir etiquetas de texto a números enteros
    etiquetas_unicas = np.unique(todas_las_etiquetas)
    etiqueta_a_entero = {etiqueta: i for i, etiqueta in enumerate(etiquetas_unicas)}
    etiquetas_enteras = np.array([etiqueta_a_entero[etiqueta] for etiqueta in todas_las_etiquetas])

    # Aplicar One-hot encoding a las etiquetas numéricas
    etiquetas_one_hot = to_categorical(etiquetas_enteras, num_classes=len(etiquetas_unicas))

    # Apilar todas las características extraídas en una única matriz NumPy
    # La función 'extract_features_song' ya se encarga de asegurar que todas las características tengan la misma forma
    caracteristicas_finales = np.stack(todas_las_caracteristicas)

    return caracteristicas_finales, etiquetas_one_hot, etiquetas_unicas

# Generar las características y etiquetas llamando a la función definida anteriormente
# Asegúrate de que la ruta 'Data/genres_original' sea correcta en tu entorno
caracteristicas_finales, etiquetas_one_hot, etiquetas_unicas = generate_features_and_labels('Data/genres_original')


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# train_test_split baraja (shuffle=True por defecto) y divide todo de una vez
# Usamos caracteristicas_finales directamente, ya que ahora es 2D
X_train, X_test, y_train, y_test = train_test_split(
    caracteristicas_finales, 
    etiquetas_one_hot, 
    test_size=0.2, 
    random_state=42
)

# --- NUEVO: Escalar los datos globalmente ---
scaler = StandardScaler()
# El escalador "aprende" la media y varianza solo del conjunto de entrenamiento
X_train = scaler.fit_transform(X_train)
# Y aplicamos esa misma escala al conjunto de test
X_test = scaler.transform(X_test)

print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de y_train: {y_train.shape}")

# Redimensionar X para que coincida con la entrada esperada de la red si es necesario
# (En este caso, lo mantenemos plano para una capa Dense inicial)
#print(f"Forma de X_train: {X_train.shape}")
#print(f"Forma de y_train: {y_train.shape}")

######################################################### PARTE 3  Análisis del modelo (entrenamiento de la red neuronal) ########################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Supongamos que ya tienes tus datos cargados: X_train, y_train, X_test, y_test
# El tamaño de entrada se obtiene de las columnas de tu dataset (X_train.shape[1])
input_dim = X_train.shape[1] 
# --- 1. Construcción del modelo ---
model = Sequential()
# Primera capa densa: 100 neuronas, activación 'relu' y tamaño de entrada
model.add(Dense(100, activation='relu', input_shape=(input_dim,)))
# Segunda capa densa: 10 neuronas (para los 10 géneros), activación 'softmax'
model.add(Dense(10, activation='softmax'))
# --- 2. Compilación del modelo ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', # Asegúrate de que tus etiquetas estén en formato one-hot
    metrics=['accuracy']
)
# Mostrar el resumen de la arquitectura
model.summary()
# --- 3. Entrenamiento del modelo ---
history = model.fit(
    X_train, 
    y_train, 
    epochs=500, 
    batch_size=32, 
    validation_split=0.2
)
# --- 4. Evaluación del sistema ---
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nPrecisión en el conjunto de test: {accuracy:.4f}")