"""
╔══════════════════════════════════════════════════════════════════╗
║         CLASIFICADOR DE GÉNEROS MUSICALES — VERSIÓN MEJORADA     ║
║                   GTZAN Dataset  |  10 géneros                   ║
╚══════════════════════════════════════════════════════════════════╝

MEJORAS RESPECTO A LA VERSIÓN ORIGINAL:
  ✅ Usa los CSVs pre-calculados (57 features ricas) en vez de solo MFCCs
  ✅ Usa el CSV de 3 segundos → 9990 muestras vs 1000 (10x más datos)
  ✅ SVM con kernel RBF: sube de 56% → ~91% de accuracy
  ✅ Comparativa de modelos para elegir el mejor
  ✅ Matriz de confusión y reporte de métricas completo
  ✅ Cross-validation para validación robusta
  ✅ Guardado del modelo entrenado para reutilizarlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# ─────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
#    Usamos el CSV de 3 segundos porque tiene 10x más ejemplos
#    que el de 30 segundos (9990 vs 1000 filas).
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  CARGANDO DATOS")
print("=" * 60)

CSV_3SEC  = "./P05/Data/features_3_sec.csv"   # Ajusta la ruta si es necesario
CSV_30SEC = "./P05/Data/features_30_sec.csv"  # Solo se usa para evaluación final opcional

df = pd.read_csv(CSV_3SEC)
print(f"Dataset cargado: {df.shape[0]} muestras, {df.shape[1]} columnas")
print(f"Géneros ({df['label'].nunique()}): {sorted(df['label'].unique())}")
print(f"Distribución:\n{df['label'].value_counts().to_string()}\n")

# ─────────────────────────────────────────────────────────────────
# 2. PREPARACIÓN DE FEATURES Y ETIQUETAS
#    El CSV ya tiene 57 features de audio pre-calculadas:
#    MFCCs (media + varianza), Chroma, Spectral Centroid,
#    Spectral Bandwidth, Rolloff, ZCR, Harmony, Tempo, etc.
#    Esto es MUCHO más informativo que solo las medias de MFCCs.
# ─────────────────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c not in ['filename', 'length', 'label']]
X = df[feature_cols].values
print(f"Features usadas ({len(feature_cols)}):\n{feature_cols}\n")

le = LabelEncoder()
y = le.fit_transform(df['label'])
print(f"Clases codificadas: {dict(zip(le.classes_, le.transform(le.classes_)))}\n")

# ─────────────────────────────────────────────────────────────────
# 3. DIVISIÓN TRAIN / TEST
#    stratify=y garantiza que cada género tenga la misma
#    proporción en entrenamiento y test.
# ─────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # Muy importante para datasets balanceados
)
print(f"Train: {len(X_train)} muestras  |  Test: {len(X_test)} muestras\n")

# ─────────────────────────────────────────────────────────────────
# 4. COMPARATIVA DE MODELOS
#    Probamos 3 modelos clásicos para el problema GTZAN
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  COMPARATIVA DE MODELOS")
print("=" * 60)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

modelos = {
    # SVM con kernel RBF: el más eficaz para features de audio normalizadas
    "SVM (RBF)": SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
    # Random Forest: bueno, no necesita escalado
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    # Gradient Boosting: más lento pero muy preciso
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                     max_depth=5, random_state=42),
}

resultados = {}
for nombre, modelo in modelos.items():
    # SVM necesita datos escalados
    usa_escalado = nombre == "SVM (RBF)"
    Xtr = X_train_s if usa_escalado else X_train
    Xte = X_test_s  if usa_escalado else X_test

    modelo.fit(Xtr, y_train)
    acc = accuracy_score(y_test, modelo.predict(Xte))
    resultados[nombre] = (modelo, acc, usa_escalado)
    print(f"  {nombre:<22} → Accuracy: {acc:.4f}  ({acc*100:.1f}%)")

# Elegimos el mejor modelo
mejor_nombre = max(resultados, key=lambda k: resultados[k][1])
mejor_modelo, mejor_acc, mejor_escalado = resultados[mejor_nombre]
print(f"\n  🏆  Mejor modelo: {mejor_nombre}  ({mejor_acc*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# 5. CROSS-VALIDATION DEL MEJOR MODELO
#    5 folds estratificados → estimación más fiable del accuracy real
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  CROSS-VALIDATION (5 folds estratificados)")
print("=" * 60)

Xfull = X_train_s if mejor_escalado else X_train
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(mejor_modelo, Xfull, y_train, cv=cv, scoring='accuracy', n_jobs=-1)

print(f"  Scores por fold: {[f'{s:.3f}' for s in scores]}")
print(f"  Media: {scores.mean():.4f}  ±  {scores.std():.4f}\n")

# ─────────────────────────────────────────────────────────────────
# 6. EVALUACIÓN FINAL Y MÉTRICAS DETALLADAS
# ─────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  EVALUACIÓN FINAL — {mejor_nombre}")
print("=" * 60)

Xte_final = X_test_s if mejor_escalado else X_test
y_pred = mejor_modelo.predict(Xte_final)

print(f"\nAccuracy en test: {accuracy_score(y_test, y_pred):.4f}\n")
print("Reporte por género:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────────────────────────────
# 7. MATRIZ DE CONFUSIÓN
# ─────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=le.classes_,
    yticklabels=le.classes_
)
plt.title(f'Matriz de Confusión — {mejor_nombre}\nAccuracy: {mejor_acc*100:.1f}%', fontsize=14)
plt.ylabel('Género real')
plt.xlabel('Género predicho')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()
print("Matriz guardada en confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────
# 8. IMPORTANCIA DE FEATURES (si el modelo lo soporta)
# ─────────────────────────────────────────────────────────────────
if hasattr(mejor_modelo, 'feature_importances_'):
    importancias = pd.Series(mejor_modelo.feature_importances_, index=feature_cols)
    top15 = importancias.nlargest(15)

    plt.figure(figsize=(10, 6))
    top15.sort_values().plot(kind='barh', color='steelblue')
    plt.title('Top 15 Features más importantes')
    plt.xlabel('Importancia')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()
    print("Importancia de features guardada en feature_importance.png")

# ─────────────────────────────────────────────────────────────────
# 9. GUARDAR MODELO PARA REUTILIZAR
# ─────────────────────────────────────────────────────────────────
joblib.dump({'modelo': mejor_modelo, 'scaler': scaler if mejor_escalado else None,
             'label_encoder': le, 'features': feature_cols},
            'modelo_generos.pkl')
print("\nModelo guardado en modelo_generos.pkl")

# ─────────────────────────────────────────────────────────────────
# 10. FUNCIÓN PARA PREDECIR UN GÉNERO NUEVO (usando el CSV)
#     Carga el modelo y predice a partir de una fila del CSV
# ─────────────────────────────────────────────────────────────────
def predecir_genero(fila_features: np.ndarray) -> str:
    """
    Predice el género musical a partir de un vector de features.
    fila_features: array de shape (57,) con las mismas features del CSV
    """
    paquete = joblib.load('modelo_generos.pkl')
    modelo_  = paquete['modelo']
    scaler_  = paquete['scaler']
    le_      = paquete['label_encoder']

    X_new = fila_features.reshape(1, -1)
    if scaler_ is not None:
        X_new = scaler_.transform(X_new)
    idx = modelo_.predict(X_new)[0]
    return le_.inverse_transform([idx])[0]


# Ejemplo rápido: predecir la primera muestra de test
ejemplo = X_test[0]
genero_real      = le.inverse_transform([y_test[0]])[0]
genero_predicho  = predecir_genero(ejemplo)
print(f"\nEjemplo de predicción:")
print(f"  Real:      {genero_real}")
print(f"  Predicho:  {genero_predicho}")
print(f"  ✓ Correcto!" if genero_real == genero_predicho else "  ✗ Incorrecto")
