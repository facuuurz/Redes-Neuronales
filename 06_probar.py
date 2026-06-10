# -*- coding: utf-8 -*-
"""
Demo interactiva: escribi una resena de pelicula EN INGLES y el modelo
te dice si es positiva o negativa (con su nivel de confianza).
No es parte de la consigna, es para jugar con el modelo entrenado.

Uso:  python 06_probar.py
      (escribi una resena y Enter; vacio o Ctrl+C para salir)
"""
import pickle
from tensorflow import keras

with open("vectorizador.pkl", "rb") as f:
    vec = pickle.load(f)
model = keras.models.load_model("modelo_base.keras")

EJEMPLOS = [
    "a masterpiece, one of the best films I have ever seen",
    "boring, predictable and a complete waste of time",
]

def clasificar(texto):
    prob = float(model.predict(vec.transform([texto]).toarray(), verbose=0)[0, 0])
    etiqueta = "POSITIVA" if prob >= 0.5 else "NEGATIVA"
    confianza = prob if prob >= 0.5 else 1 - prob
    return etiqueta, confianza

print("Ejemplos:")
for t in EJEMPLOS:
    et, conf = clasificar(t)
    print(f"  [{et} {conf:.0%}] {t}")

print("\nEscribi tu resena en ingles (Enter vacio para salir):")
while True:
    try:
        texto = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not texto:
        break
    et, conf = clasificar(texto)
    print(f"  -> {et} (confianza {conf:.0%})")
