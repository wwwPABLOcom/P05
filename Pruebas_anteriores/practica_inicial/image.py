import librosa
import librosa.display
import glob
import numpy as np
import matplotlib.pyplot as plt

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

if __main__ == "__name__":
    # Asumiendo que has descargado los archivos y están en tu carpeta:
    mfcc_kick = display_mfcc('kick_loop.wav', 'Kick Loop (Bajos)')
    mfcc_whistle = display_mfcc('whistling.wav', 'Whistling (Agudos)')