# Importaciones de Keras para la Red Neuronal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical

# Ejemplo de estructura del modelo
model = Sequential()

# Primera capa Densa
# Supongamos que pasamos la media de 13 MFCCs como entrada
model.add(Dense(64, input_shape=(13,))) 
model.add(Activation('relu'))

# Segunda capa Densa
model.add(Dense(32))
model.add(Activation('relu'))

# Capa de salida
# Supongamos 10 géneros musicales diferentes
model.add(Dense(10))
model.add(Activation('softmax')) # Softmax para clasificación multiclase

# Ver el resumen de la arquitectura
model.summary()