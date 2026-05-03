import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. Define las rutas de tus archivos (asegúrate de poner el nombre exacto de tus descargas)
ruta_kick = '266093__stereo-surgeon__kick-loop-5.wav' 
ruta_whistle = '98195__grrlrighter__whistling.wav'

# 2. Cargar los archivos de audio
# El parámetro sr=None preserva la frecuencia de muestreo original del archivo
y_kick, sr_kick = librosa.load(ruta_kick, sr=None)
y_whistle, sr_whistle = librosa.load(ruta_whistle, sr=None)

# 3. Calcular los MFCCs
# n_mfcc=13 es el estándar de la industria para capturar la envoltura básica del sonido
mfcc_kick = librosa.feature.mfcc(y=y_kick, sr=sr_kick, n_mfcc=13)
mfcc_whistle = librosa.feature.mfcc(y=y_whistle, sr=sr_whistle, n_mfcc=13)

# 4. Visualizar y comparar los resultados
plt.figure(figsize=(12, 8))

# Gráfico para el Kick Loop
plt.subplot(2, 1, 1)
librosa.display.specshow(mfcc_kick, x_axis='time', sr=sr_kick, cmap='viridis')
plt.colorbar(format='%+2.0f')
plt.title('Valores MFCC - Kick Loop (Bajas Frecuencias / Percusivo)')
plt.ylabel('Coeficientes MFCC')

# Gráfico para el Silbido
plt.subplot(2, 1, 2)
librosa.display.specshow(mfcc_whistle, x_axis='time', sr=sr_whistle, cmap='viridis')
plt.colorbar(format='%+2.0f')
plt.title('Valores MFCC - Whistling (Altas Frecuencias / Tonal)')
plt.ylabel('Coeficientes MFCC')
plt.xlabel('Tiempo')

plt.tight_layout()
plt.show()