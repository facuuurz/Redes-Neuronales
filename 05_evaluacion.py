# -*- coding: utf-8 -*-
"""
TP Redes Neuronales - Punto 6: Matriz de confusion y metricas
Evalua el modelo base sobre el conjunto de TEST (nunca visto en
entrenamiento ni en seleccion de hiperparametros).
Calcula accuracy, precision, recall y F1, y grafica la matriz de confusion.
"""
import pickle
import numpy as np
from tensorflow import keras
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("data.pkl", "rb") as f:
    X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

model = keras.models.load_model("modelo_base.keras")
y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("=== Matriz de confusion (test) ===")
print("                Pred NEG  Pred POS")
print(f"  Real NEG (0):   {cm[0,0]:5d}     {cm[0,1]:5d}")
print(f"  Real POS (1):   {cm[1,0]:5d}     {cm[1,1]:5d}")
print(f"\nAccuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print("""
Nota: el dataset esta perfectamente balanceado (50% pos / 50% neg) y los
errores en ambas clases cuestan lo mismo, por lo que ACCURACY es una
metrica adecuada. F1 se reporta como complemento.""")

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=["Negativa", "Positiva"]).plot(
    ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Matriz de confusion (test)\naccuracy={acc:.3f}  f1={f1:.3f}")
plt.tight_layout()
plt.savefig("fig_matriz_confusion.png", dpi=150)
print("Grafico guardado: fig_matriz_confusion.png")

with open("metricas_test.txt", "w") as f:
    f.write(f"accuracy={acc:.4f}\nprecision={prec:.4f}\nrecall={rec:.4f}\nf1={f1:.4f}\n")
    f.write(f"matriz_confusion=\n{cm}\n")
