# -*- coding: utf-8 -*-
"""
TP Redes Neuronales - Punto 2: Dataset de Hugging Face
Dataset: rotten_tomatoes (critica de cine, clasificacion binaria pos/neg)
https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes

Descarga el dataset, lo explora y guarda los textos vectorizados (TF-IDF)
para que los demas scripts entrenen rapido.
"""
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

print("Descargando dataset rotten_tomatoes...")
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")

print("\n=== Estructura del dataset ===")
print(ds)
print("\n=== Ejemplos ===")
for i in [0, 5000]:
    print(f"  label={ds['train'][i]['label']} | {ds['train'][i]['text'][:80]}...")

print("\n=== Balance de clases (train) ===")
labels = np.array(ds["train"]["label"])
print(f"  Negativas (0): {(labels == 0).sum()}")
print(f"  Positivas (1): {(labels == 1).sum()}")

# Vectorizacion TF-IDF: convierte cada resena en un vector de 10000 features
# (frecuencia de palabras ponderada). Entrada apta para un MLP denso.
print("\nVectorizando con TF-IDF (max 10000 features)...")
vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train = vec.fit_transform(ds["train"]["text"]).toarray().astype("float32")
X_val   = vec.transform(ds["validation"]["text"]).toarray().astype("float32")
X_test  = vec.transform(ds["test"]["text"]).toarray().astype("float32")
y_train = np.array(ds["train"]["label"], dtype="float32")
y_val   = np.array(ds["validation"]["label"], dtype="float32")
y_test  = np.array(ds["test"]["label"], dtype="float32")

print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

with open("data.pkl", "wb") as f:
    pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)
# El vectorizador se guarda aparte para poder clasificar texto nuevo (06_probar.py)
with open("vectorizador.pkl", "wb") as f:
    pickle.dump(vec, f)
print("\nGuardado en data.pkl y vectorizador.pkl")
