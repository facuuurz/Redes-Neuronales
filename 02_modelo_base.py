# -*- coding: utf-8 -*-
"""
TP Redes Neuronales - Punto 3: Red neuronal (MLP) con Keras Sequential
Modelo base: 2 capas ocultas con dropout, salida sigmoide para
clasificacion binaria.
"""
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tf.random.set_seed(42)
np.random.seed(42)

with open("data.pkl", "rb") as f:
    X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)

# Modelo secuencial de Keras: las capas se apilan en orden
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),  # salida binaria
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

# EarlyStopping: corta el entrenamiento cuando la val_loss deja de mejorar
# y restaura los pesos de la mejor epoca (evita el overfitting del punto 4)
early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                      restore_best_weights=True)
hist = model.fit(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=15, batch_size=64, verbose=2,
                 callbacks=[early])

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n=== Modelo base -> test accuracy: {test_acc:.4f}, test loss: {test_loss:.4f} ===")

model.save("modelo_base.keras")

# Curvas de entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(hist.history["loss"], label="train")
axes[0].plot(hist.history["val_loss"], label="validacion")
axes[0].set_title("Loss - modelo base")
axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("Binary crossentropy"); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].plot(hist.history["accuracy"], label="train")
axes[1].plot(hist.history["val_accuracy"], label="validacion")
axes[1].set_title("Accuracy - modelo base")
axes[1].set_xlabel("Epoca"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig("fig_modelo_base.png", dpi=150)
print("Grafico guardado: fig_modelo_base.png")
