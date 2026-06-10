# -*- coding: utf-8 -*-
"""
TP Redes Neuronales - Punto 4: Simular overfitting
Red grande (512-256-128), sin dropout ni regularizacion, muchas epocas.
La loss de train baja a casi 0 mientras la de validacion sube:
el modelo memoriza el train y pierde capacidad de generalizar.
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

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation="relu"),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy"])

hist = model.fit(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=40, batch_size=64, verbose=2)

h = hist.history
best_ep = int(np.argmin(h["val_loss"])) + 1
print(f"\n=== Overfitting ===")
print(f"Mejor val_loss en epoca {best_ep}: {min(h['val_loss']):.4f}")
print(f"val_loss final (epoca 40): {h['val_loss'][-1]:.4f}")
print(f"train_loss final: {h['loss'][-1]:.4f} (casi 0 -> memorizo el train)")
print(f"train_acc final: {h['accuracy'][-1]:.4f} | val_acc final: {h['val_accuracy'][-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(h["loss"], label="train")
axes[0].plot(h["val_loss"], label="validacion")
axes[0].axvline(best_ep - 1, color="gray", ls="--", label=f"mejor epoca ({best_ep})")
axes[0].set_title("Overfitting: la val_loss sube mientras train_loss baja")
axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("Binary crossentropy"); axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].plot(h["accuracy"], label="train")
axes[1].plot(h["val_accuracy"], label="validacion")
axes[1].set_title("Accuracy: train llega a ~100%, validacion se estanca")
axes[1].set_xlabel("Epoca"); axes[1].set_ylabel("Accuracy"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout()
plt.savefig("fig_overfitting.png", dpi=150)
print("Grafico guardado: fig_overfitting.png")
