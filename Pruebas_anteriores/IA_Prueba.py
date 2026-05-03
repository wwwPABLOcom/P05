import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

########################################################## 
# PARTE 1 y 2: Carga y Preprocesado instantáneo (usando el CSV)
########################################################## 

print("Cargando datos del CSV...")
# 1. Leemos el archivo (Asegúrate de que la ruta sea correcta)
df = pd.read_csv('Data/features_3_sec.csv')

# 2. Separamos las características (X) de las etiquetas (y)
# Quitamos las columnas que no son matemáticas ('filename' y 'length') y la de la etiqueta ('label')
X = df.drop(columns=['filename', 'length', 'label']).values

# Cogemos solo la columna de la etiqueta (el género musical)
etiquetas_texto = df['label'].values

# 3. Convertimos las etiquetas de texto a números y luego a One-Hot
encoder = LabelEncoder()
etiquetas_numeros = encoder.fit_transform(etiquetas_texto)
y = to_categorical(etiquetas_numeros)

# 4. Dividimos en Train y Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Escalamos los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Forma de X_train: {X_train.shape} (¡Mira cuántos datos y características!)")

######################################################### 
# PARTE 3: Entrenamiento de la red neuronal mejorada
######################################################### 

input_dim = X_train.shape[1] 

model = Sequential()
model.add(Input(shape=(input_dim,)))

# Usamos la red profunda con Dropout para evitar que memorice
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

# Capa de salida: 10 géneros
model.add(Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Entrenando el modelo...")
history = model.fit(
    X_train, 
    y_train, 
    epochs=200, # 100 épocas son suficientes con tantos datos
    batch_size=32, 
    validation_split=0.2
)

# Evaluación
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nPrecisión en el conjunto de test: {accuracy:.4f}")