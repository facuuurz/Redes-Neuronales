# TP - Redes Neuronales (MLP con Keras)

Clasificación binaria de sentimiento sobre el dataset
[rotten_tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
(reseñas de cine, 50% positivas / 50% negativas).

## Requisitos

```
pip install tensorflow-cpu datasets scikit-learn matplotlib pandas
```

## Cómo correr (en orden)

| Script | Punto del TP | Salida |
|---|---|---|
| `01_dataset.py` | 2 — Dataset de Hugging Face | `data.pkl` (TF-IDF, 10k features), `vectorizador.pkl` |
| `02_modelo_base.py` | 3 — Red neuronal (Keras Sequential) | `modelo_base.keras`, `fig_modelo_base.png` |
| `03_overfitting.py` | 4 — Simular overfitting | `fig_overfitting.png` |
| `04_experimentos.py` | 5 — 4 experimentos de hiperparámetros | `experimentos.csv`, `fig_experimentos.png` |
| `05_evaluacion.py` | 6 — Matriz de confusión y métricas | `fig_matriz_confusion.png`, `metricas_test.txt` |

```
python 01_dataset.py
python 02_modelo_base.py
python 03_overfitting.py
python 04_experimentos.py
python 05_evaluacion.py
```

> `data.pkl` (407 MB) no está en el repo: se regenera corriendo `01_dataset.py`.

## Probar el modelo con texto propio

```
python 06_probar.py
```

Demo interactiva (no es parte de la consigna): escribís una reseña de película
**en inglés** y el modelo responde POSITIVA o NEGATIVA con su confianza.
Requiere haber corrido antes `01_dataset.py` y `02_modelo_base.py`
(el modelo entrenado `modelo_base.keras` ya está incluido en el repo).

## Resultados

Los números, tablas y observaciones de todos los experimentos están
organizados por punto de la consigna en [`RESULTADOS.md`](RESULTADOS.md).
Resumen: el modelo base logra **78,05% de accuracy** sobre test (dataset
balanceado 50/50, azar = 50%).

## Diseño

- **Vectorización**: TF-IDF con 10.000 features (unigramas + bigramas). Un MLP
  denso necesita entrada de tamaño fijo; TF-IDF es la opción clásica para texto.
- **Modelo base**: `Dense(64) → Dropout(0.5) → Dense(32) → Dropout(0.5) → Dense(1, sigmoid)`,
  Adam lr=1e-3, binary crossentropy, 15 épocas, batch 64.
- **Overfitting**: red grande (512-256-128) sin dropout, 40 épocas → la
  val_loss sube mientras la train_loss baja a ~0.
- **Experimentos**: learning rate, tamaño de capas, dropout y batch size,
  variando uno a la vez sobre la config base.
- **Métrica**: el dataset está balanceado → accuracy es adecuada; se reporta
  también F1, precision y recall.

> Nota: el informe PDF (máx. 5 páginas) debe redactarse a mano — la consigna
> prohíbe usar IA para escribir el informe.
