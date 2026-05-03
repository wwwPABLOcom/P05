"""
╔══════════════════════════════════════════════════════════════════╗
║       CLASIFICADOR DE GÉNEROS — CNN CON TRANSFER LEARNING        ║
║         MobileNetV2  |  10 géneros  |  Espectrogramas Mel        ║
╚══════════════════════════════════════════════════════════════════╝

POR QUÉ CNN Y NO SVM:
  - El SVM aprendía features numéricas fijas → no generaliza bien
  - La CNN aprende patrones visuales del espectrograma → generaliza
  - Con Transfer Learning (MobileNetV2 preentrenada en ImageNet)
    conseguimos buen accuracy con solo ~100 imágenes por género

REQUISITOS:
    pip install tensorflow librosa pillow numpy matplotlib scikit-learn

ESTRUCTURA DE CARPETAS ESPERADA:
    images_original/
        blues/       blues00000.png  blues00001.png  ...
        classical/   classical00000.png ...
        country/     ...
        disco/
        hiphop/
        jazz/
        metal/
        pop/
        reggae/
        rock/

USO:
    1. Pon este script en la misma carpeta que images_original/
    2. python cnn_generos_musical.py                     → entrena
    3. python cnn_generos_musical.py mi_cancion.wav      → predice
"""

import os
import sys
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Sin ventana gráfica (compatible con servidores)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────
IMAGES_DIR   = "./P05/Data/images_original/"      # Carpeta con los espectrogramas
MODELO_PATH  = "modelo_cnn_generos.keras"
IMG_SIZE     = (224, 224)             # MobileNetV2 espera 224x224
BATCH_SIZE   = 16                     # Pequeño porque tenemos pocas imágenes
EPOCHS       = 30                     # Suficiente con transfer learning
GENEROS      = ['blues', 'classical', 'country', 'disco', 'hiphop',
                'jazz', 'metal', 'pop', 'reggae', 'rock']


# ══════════════════════════════════════════════════════════════════
# PARTE 1: PREPARACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════

def cargar_dataset(images_dir: str):
    """
    Carga todas las imágenes PNG organizadas por carpeta de género.
    Devuelve arrays X (imágenes) e y (etiquetas numéricas).
    """
    import tensorflow as tf

    print("=" * 55)
    print("  CARGANDO DATASET DE ESPECTROGRAMAS")
    print("=" * 55)

    # Usamos ImageDataGenerator de Keras — carga y redimensiona automáticamente
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Data augmentation para el entrenamiento (aumenta variedad artificialmente)
    # Muy importante con datasets pequeños (~100 imágenes por clase)
    datagen_train = ImageDataGenerator(
        rescale=1.0 / 255.0,          # Normalizar píxeles a [0, 1]
        validation_split=0.2,          # 20% para validación
        rotation_range=5,              # Rotación leve (espectrogramas son sensibles)
        width_shift_range=0.1,         # Desplazamiento horizontal
        zoom_range=0.05,               # Zoom muy suave
        horizontal_flip=False,         # NO voltear — cambia el significado temporal
    )

    datagen_val = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
    )

    train_gen = datagen_train.flow_from_directory(
        images_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=42,
        color_mode='rgb',
    )

    val_gen = datagen_val.flow_from_directory(
        images_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=42,
        color_mode='rgb',
    )

    print(f"\n  Clases detectadas: {train_gen.class_indices}")
    print(f"  Imágenes entrenamiento: {train_gen.samples}")
    print(f"  Imágenes validación:    {val_gen.samples}\n")

    return train_gen, val_gen, train_gen.class_indices


# ══════════════════════════════════════════════════════════════════
# PARTE 2: CONSTRUCCIÓN DEL MODELO CNN
# ══════════════════════════════════════════════════════════════════

def construir_modelo(num_clases: int = 10):
    """
    Construye una CNN usando Transfer Learning con MobileNetV2.

    Arquitectura:
        MobileNetV2 (preentrenada en ImageNet, pesos congelados)
            ↓
        GlobalAveragePooling2D
            ↓
        Dense(256, relu) + Dropout(0.5)
            ↓
        Dense(128, relu) + Dropout(0.3)
            ↓
        Dense(10, softmax)  ← salida: probabilidad por género

    Por qué MobileNetV2:
        - Ligera y rápida de entrenar (importante con pocos datos)
        - Preentrenada en 1M imágenes → ya sabe detectar texturas y patrones
        - Solo entrenamos las capas finales, no toda la red
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2

    print("=" * 55)
    print("  CONSTRUYENDO MODELO CNN (MobileNetV2)")
    print("=" * 55)

    # Base preentrenada — include_top=False quita la cabeza de clasificación
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # FASE 1: Congelar la base — solo entrenamos las capas nuevas
    base_model.trainable = False
    print(f"  Base MobileNetV2: {len(base_model.layers)} capas (congeladas)")

    # Cabeza de clasificación personalizada
    modelo = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),                    # Evita overfitting
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_clases, activation='softmax'),
    ])

    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    modelo.summary()
    return modelo, base_model


# ══════════════════════════════════════════════════════════════════
# PARTE 3: ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════

def entrenar_modelo(images_dir: str = IMAGES_DIR, guardar_en: str = MODELO_PATH):
    """
    Entrena la CNN en dos fases:
      Fase 1 — Solo las capas nuevas (base congelada) → convergencia rápida
      Fase 2 — Fine-tuning: descongelar últimas capas de la base → más precisión
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Cargar datos
    train_gen, val_gen, class_indices = cargar_dataset(images_dir)
    num_clases = len(class_indices)

    # Construir modelo
    modelo, base_model = construir_modelo(num_clases)

    # Callbacks — paran el entrenamiento si deja de mejorar
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        verbose=1
    )

    # ── FASE 1: Entrenamiento de la cabeza ──────────────────────
    print("\n" + "=" * 55)
    print("  FASE 1: Entrenando cabeza de clasificación")
    print("=" * 55)

    history1 = modelo.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ── FASE 2: Fine-tuning de las últimas capas ────────────────
    print("\n" + "=" * 55)
    print("  FASE 2: Fine-tuning (últimas 30 capas de MobileNetV2)")
    print("=" * 55)

    # Descongelar solo las últimas 30 capas de la base
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    # Learning rate más pequeño para no destruir los pesos aprendidos
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop2 = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    history2 = modelo.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=[early_stop2, reduce_lr],
        verbose=1
    )

    # Guardar modelo y mapeo de clases
    modelo.save(guardar_en)

    # Guardar también el mapeo clase→índice para predecir después
    import json
    idx_to_class = {v: k for k, v in class_indices.items()}
    with open('clases_generos.json', 'w') as f:
        json.dump(idx_to_class, f)

    print(f"\n  💾 Modelo guardado en: {guardar_en}")
    print(f"  💾 Clases guardadas en: clases_generos.json")

    # Gráfica de entrenamiento
    _plot_history(history1, history2)

    return modelo, idx_to_class


def _plot_history(h1, h2):
    """Genera gráfica de accuracy y loss durante el entrenamiento."""
    acc  = h1.history['accuracy']  + h2.history['accuracy']
    val  = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    v_loss = h1.history['val_loss'] + h2.history['val_loss']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(acc,  label='Train accuracy')
    ax1.plot(val,  label='Val accuracy')
    ax1.axvline(len(h1.history['accuracy']) - 1, color='gray',
                linestyle='--', label='Inicio fine-tuning')
    ax1.set_title('Accuracy por época')
    ax1.set_xlabel('Época')
    ax1.legend()

    ax2.plot(loss,   label='Train loss')
    ax2.plot(v_loss, label='Val loss')
    ax2.axvline(len(h1.history['loss']) - 1, color='gray',
                linestyle='--', label='Inicio fine-tuning')
    ax2.set_title('Loss por época')
    ax2.set_xlabel('Época')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('entrenamiento_cnn.png', dpi=150)
    print("  📊 Gráfica guardada en: entrenamiento_cnn.png")


# ══════════════════════════════════════════════════════════════════
# PARTE 4: CONVERTIR WAV → ESPECTROGRAMA MEL
# ══════════════════════════════════════════════════════════════════

def wav_a_espectrograma(ruta_audio: str) -> np.ndarray:
    """
    Convierte un archivo .wav o .mp3 en un espectrograma Mel
    del mismo formato que las imágenes de entrenamiento.

    Devuelve array numpy de shape (224, 224, 3) normalizado [0,1].
    """
    import librosa
    import librosa.display
    from PIL import Image
    import io

    # Cargar audio (máximo 30 segundos, igual que GTZAN)
    y, sr = librosa.load(ruta_audio, duration=30, mono=True)

    # Generar espectrograma Mel (mismos parámetros que GTZAN)
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=128,
        fmax=8000
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Renderizar como imagen (igual que las PNGs de entrenamiento)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=72)
    fig.subplots_adjust(0, 0, 1, 1)  # Sin márgenes
    ax.axis('off')
    librosa.display.specshow(mel_db, sr=sr, fmax=8000, ax=ax, cmap='viridis')

    # Capturar imagen en memoria (sin guardar a disco)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Redimensionar a 224x224 y normalizar
    img = Image.open(buf).convert('RGB')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0

    return arr


# ══════════════════════════════════════════════════════════════════
# PARTE 5: PREDICCIÓN DE GÉNERO
# ══════════════════════════════════════════════════════════════════

def predecir_cancion(ruta_audio: str,
                     modelo_path: str = MODELO_PATH) -> dict:
    """
    Predice el género musical de cualquier .wav o .mp3.

    Uso:
        resultado = predecir_cancion("mi_cancion.wav")
        print(resultado['genero'], resultado['confianza'])
    """
    import tensorflow as tf
    import json

    if not os.path.exists(ruta_audio):
        raise FileNotFoundError(f"No se encuentra: {ruta_audio}")

    # Entrenar si no existe el modelo
    if not os.path.exists(modelo_path):
        print("Modelo no encontrado. Entrenando primero...")
        if not os.path.exists(IMAGES_DIR):
            raise FileNotFoundError(
                f"No se encuentra la carpeta '{IMAGES_DIR}'. "
                "Descomprime images_original.zip en la misma carpeta que este script."
            )
        entrenar_modelo()

    # Cargar modelo y clases
    modelo = tf.keras.models.load_model(modelo_path)
    with open('clases_generos.json') as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}

    # Convertir audio a espectrograma
    print(f"\n  Analizando: {os.path.basename(ruta_audio)}...")
    espectrograma = wav_a_espectrograma(ruta_audio)
    X = np.expand_dims(espectrograma, axis=0)  # (1, 224, 224, 3)

    # Predecir
    probas   = modelo.predict(X, verbose=0)[0]
    ranking  = sorted(enumerate(probas), key=lambda x: x[1], reverse=True)
    top3     = [(idx_to_class[i], round(float(p) * 100, 1)) for i, p in ranking[:3]]

    resultado = {
        'genero':    top3[0][0],
        'confianza': top3[0][1],
        'top3':      top3,
        'todas':     {idx_to_class[i]: round(float(p) * 100, 1) for i, p in ranking},
    }

    # Mostrar resultado
    print("\n" + "─" * 45)
    print(f"  🎵  Género detectado: {resultado['genero'].upper()}")
    print(f"  📊  Confianza:        {resultado['confianza']}%")
    print(f"\n  Top 3 más probables:")
    for i, (g, p) in enumerate(top3):
        barra = "█" * int(p / 5)
        print(f"    {i+1}. {g:<12} {barra} {p}%")
    print("─" * 45 + "\n")

    return resultado


# ══════════════════════════════════════════════════════════════════
# EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    if len(sys.argv) > 1:
        # ── Modo predicción: python cnn_generos_musical.py cancion.wav ──
        ruta = sys.argv[1]
        predecir_cancion(ruta)

    else:
        # ── Modo entrenamiento ──
        if not os.path.exists(IMAGES_DIR):
            print(f"ERROR: No se encuentra la carpeta '{IMAGES_DIR}'")
            print("Descomprime images_original.zip aquí y vuelve a ejecutar.")
            sys.exit(1)

        if os.path.exists(MODELO_PATH):
            print(f"✅ Modelo ya entrenado en '{MODELO_PATH}'")
            print("   Bórralo y vuelve a ejecutar si quieres reentrenar.")
            print()
            print("Para predecir una canción:")
            print("   python cnn_generos_musical.py mi_cancion.wav")
        else:
            print("Iniciando entrenamiento CNN...")
            print("⚠️  Esto puede tardar 5-15 minutos según tu hardware.")
            print("    Con GPU tardará mucho menos.\n")
            entrenar_modelo()
            print("\n✅ Entrenamiento completado.")
            print("Ahora puedes predecir canciones:")
            print("   python cnn_generos_musical.py mi_cancion.wav")
