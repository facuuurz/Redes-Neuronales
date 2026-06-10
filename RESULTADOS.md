# Resultados del TP (insumos para el informe)

> La consigna prohíbe usar IA para redactar el informe — estos son los
> datos crudos y observaciones técnicas para que escribas el texto vos.

## Dataset (punto 2)

- **rotten_tomatoes** (Hugging Face): reseñas cortas de cine en inglés.
- Train 8.530 / Validación 1.066 / Test 1.066. Balance perfecto 50/50.
- Vectorización: TF-IDF, 10.000 features (unigramas + bigramas).

## Modelo base (punto 3)

`Dense(64, relu) → Dropout(0.5) → Dense(32, relu) → Dropout(0.5) → Dense(1, sigmoid)`
Adam lr=1e-3, binary crossentropy, batch 64, EarlyStopping (patience 3, restaura mejores pesos).

- Paró en la época 5, restauró pesos de la **época 2** (mejor val_loss 0.4975).
- **Test accuracy: 0.7805** | test loss: 0.4512.
- Gráfico: `fig_modelo_base.png`

## Overfitting (punto 4)

Red 512-256-128 **sin** dropout, 40 épocas (gráfico: `fig_overfitting.png`):

| Métrica | Mejor época (1) | Época 40 |
|---|---|---|
| train_loss | ~0.51 | **0.0005** |
| val_loss | 0.5116 | **2.5311** (5× peor) |
| train_acc | ~0.70 | 0.9994 |
| val_acc | 0.7561 | 0.7289 |

Observación: el modelo memoriza el train (acc ~100%) mientras la val_loss se
quintuplica → pérdida total de generalización después de la época 1.

## Experimentos de hiperparámetros (punto 5)

Config base: lr=1e-3, capas (64,32), dropout 0.5, batch 64. Se varió uno por vez,
15 épocas, se reporta la mejor época según val_loss. Tabla completa: `experimentos.csv`,
curvas: `fig_experimentos.png`.

| Hiperparámetro | Valores | Mejor | val_acc |
|---|---|---|---|
| Learning rate | 1e-4 / 1e-3 / 1e-2 | **1e-4** | 0.7561 |
| Capas ocultas | (16,8) / (64,32) / (256,128) | **(64,32)** | 0.7542 |
| Dropout | 0.0 / 0.3 / 0.5 | **0.3** | 0.7561 |
| Batch size | 16 / 64 / 256 | **256** | 0.7552 |

Observaciones para explicar:

- **Learning rate**: con 1e-4 el aprendizaje es lento y gradual (mejor época 14,
  curva de val_loss descendente todo el entrenamiento) y termina generalizando
  mejor. Con 1e-3 y 1e-2 el modelo converge en 1-2 épocas y enseguida
  sobreajusta. lr=1e-2 además oscila (pasos demasiado grandes).
- **Capas ocultas**: más capacidad ≠ mejor. La red grande (256,128) sobreajusta
  más rápido (mejor época: la 1) y rinde peor (0.743). Con TF-IDF el problema
  es casi linealmente separable, alcanza con una red chica.
- **Dropout**: sin dropout (0.0) el sobreajuste es inmediato (mejor época 1).
  Con 0.3-0.5 se retrasa el sobreajuste y mejora levemente la val_acc. El
  efecto se ve en el "gap" train-val del CSV.
- **Batch size**: batch 256 hace updates más estables/menos ruidosos y, con el
  mismo número de épocas, avanza más lento → llega más lejos antes de
  sobreajustar (mejor época 4, mejor val_loss 0.4983). Batch 16 mete mucho
  ruido y sobreajusta enseguida.
- **Patrón general**: todos los experimentos rondan 74-76% val_acc — el techo lo
  pone la representación TF-IDF (bolsa de palabras, pierde orden y contexto),
  no el tamaño de la red. La defensa principal contra el overfitting acá es
  el early stopping.

## Evaluación final en test (punto 6)

Matriz de confusión del modelo base sobre test (`fig_matriz_confusion.png`):

|              | Pred NEG | Pred POS |
|---|---|---|
| **Real NEG** | 419 (TN) | 114 (FP) |
| **Real POS** | 120 (FN) | 413 (TP) |

| Métrica | Valor |
|---|---|
| **Accuracy** | **0.7805** |
| Precision | 0.7837 |
| Recall | 0.7749 |
| F1-score | 0.7792 |

**Elección de métrica**: el dataset está perfectamente balanceado (533/533 en
test) y los dos tipos de error (FP y FN) cuestan lo mismo en este problema →
**accuracy es la métrica adecuada**. Precision, recall y F1 son casi idénticas
entre sí (los errores se reparten parejo entre clases: 114 FP vs 120 FN), lo
que confirma que el modelo no está sesgado hacia ninguna clase.
