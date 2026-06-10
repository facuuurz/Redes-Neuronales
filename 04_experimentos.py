# -*- coding: utf-8 -*-
"""
TP Redes Neuronales - Punto 5: Experimentos variando hiperparametros
Se varian 4 hiperparametros, uno a la vez, partiendo de la config base:
  1. Learning rate:      1e-4, 1e-3, 1e-2
  2. Tamano de capas:    (16,8), (64,32), (256,128)
  3. Dropout:            0.0, 0.3, 0.5
  4. Batch size:         16, 64, 256
Cada config se entrena 15 epocas y se evalua en validacion.
Resultados -> experimentos.csv + fig_experimentos.png
"""
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("data.pkl", "rb") as f:
    X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

BASE = dict(lr=1e-3, capas=(64, 32), dropout=0.5, batch=64)

def entrenar(lr, capas, dropout, batch, epochs=15, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    caps = [layers.Input(shape=(X_train.shape[1],))]
    for n in capas:
        caps.append(layers.Dense(n, activation="relu"))
        if dropout > 0:
            caps.append(layers.Dropout(dropout))
    caps.append(layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(caps)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    h = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=epochs, batch_size=batch, verbose=0).history
    best = int(np.argmin(h["val_loss"]))
    return dict(val_acc=h["val_accuracy"][best], val_loss=h["val_loss"][best],
                train_acc=h["accuracy"][best], mejor_epoca=best + 1, hist=h)

experimentos = [
    ("learning_rate", "lr",      [1e-4, 1e-3, 1e-2]),
    ("capas_ocultas", "capas",   [(16, 8), (64, 32), (256, 128)]),
    ("dropout",       "dropout", [0.0, 0.3, 0.5]),
    ("batch_size",    "batch",   [16, 64, 256]),
]

filas, curvas = [], {}
for nombre, clave, valores in experimentos:
    print(f"\n=== Experimento: {nombre} ===")
    for v in valores:
        cfg = dict(BASE); cfg[clave] = v
        r = entrenar(**cfg)
        es_base = (v == BASE[clave])
        filas.append(dict(experimento=nombre, valor=str(v),
                          val_accuracy=round(r["val_acc"], 4),
                          val_loss=round(r["val_loss"], 4),
                          train_accuracy=round(r["train_acc"], 4),
                          mejor_epoca=r["mejor_epoca"],
                          gap_train_val=round(r["train_acc"] - r["val_acc"], 4),
                          es_config_base=es_base))
        curvas[(nombre, str(v))] = r["hist"]["val_loss"]
        print(f"  {clave}={v}: val_acc={r['val_acc']:.4f} val_loss={r['val_loss']:.4f} (mejor epoca {r['mejor_epoca']})")

df = pd.DataFrame(filas)
df.to_csv("experimentos.csv", index=False)
print("\n", df.to_string(index=False))
print("\nGuardado: experimentos.csv")

# Grafico: curvas de val_loss por experimento
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (nombre, clave, valores) in zip(axes.flat, experimentos):
    for v in valores:
        ax.plot(curvas[(nombre, str(v))], label=f"{clave}={v}")
    ax.set_title(f"Variando {nombre}")
    ax.set_xlabel("Epoca"); ax.set_ylabel("val_loss"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout()
plt.savefig("fig_experimentos.png", dpi=150)
print("Grafico guardado: fig_experimentos.png")
