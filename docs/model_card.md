[⬅ Volver al README](../README.md)

# Model Card

> Documento a completar durante la fase de modelado (Bimestre 1 - prototipado / Bimestre 2 - versión final en AWS).
> Para el razonamiento detrás de cada decisión ya tomada, ver
> [`documentacion_tecnica.md`](documentacion_tecnica.md).

## Descripción del modelo

- **Arquitectura:** EfficientNet-B0 (backbone preentrenado en ImageNet, congelado)
- **Framework:** PyTorch
- **Técnica:** Transfer learning (backbone preentrenado en ImageNet + fine-tuning de capas superiores)
- **Resolución de entrada:** 224×224 — elegida en el EDA (`notebooks/01_eda.ipynb`, sección 3) para alinear con la resolución preentrenada estándar de EfficientNet-B0/MobileNetV2 sobre ImageNet, aceptando upscaling en el 11.3% de las imágenes que llegan más chicas

## Dataset

- **Fuente:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Clases:** Glioma, Meningioma, Pituitario, Sin tumor
- **Split:** 70% train / 10% val / 20% test, estratificado por clase, seed fija (`config/config.yaml`). Deduplicado antes de dividir — el EDA encontró 4.72% de duplicados exactos. El test es más grande que el val para acotar el margen de error del recall por clase (~350 imágenes por clase en lugar de ~175); con el backbone congelado, la reducción del train tiene un costo bajo.

## Métricas de evaluación

Priorizadas por contexto sanitario (un falso negativo es más costoso que un falso positivo):

| Métrica | Valor | Justificación |
|---|---|---|
| Recall / Sensibilidad | TBD | Minimizar falsos negativos (tumores no detectados) |
| F1-Score | TBD | Balance entre precisión y recall por clase |
| ROC-AUC | TBD | Capacidad de discriminación entre clases |
| Matriz de confusión | TBD | Detalle de errores por clase |

## Limitaciones

- **Composición del dataset por clase (hallazgo del EDA, sección 5):** la clase "Sin
  tumor" proviene en su totalidad de una fuente distinta (Br35H) a las tres clases con
  tumor (glioma: figshare; meningioma/pituitario: mayormente SARTAJ). El EDA detectó
  diferencias sistemáticas de brillo, contraste y fondo entre clases que coinciden con
  esta composición, no con una diferencia clínica confirmada. Existe riesgo de que el
  modelo aprenda a distinguir por el "estilo" de la fuente de origen en vez de por
  patología real. Un clasificador que solo recibe el brillo medio y el desvío de cada
  imagen alcanza 63% de accuracy (azar: 25%); el augmentation de brillo/contraste lo
  reduce apenas a 61%, y la normalización por imagen a 56%. Es una limitación estructural
  del dataset que el preprocesamiento no puede eliminar, y que la métrica de test no
  detecta porque test proviene de las mismas fuentes. Se evalúa aparte con una prueba de
  estrés de brillo sobre el modelo entrenado (detalle en la
  [documentación técnica](documentacion_tecnica.md)).
- **Tipo de secuencia de resonancia confundido con la clase:** las clases con tumor son,
  casi en su totalidad, imágenes T1 con contraste, mientras que "Sin tumor" mezcla T1,
  T2 y FLAIR (identificación visual sobre una muestra, pendiente de confirmación
  médica). El modelo puede asociar el aspecto T2/FLAIR con "Sin tumor", lo que ante una
  imagen T2 con tumor produciría un falso negativo. El augmentation de intensidad no
  corrige esta diferencia. Se evaluará con un etiquetado manual de secuencias sobre una
  muestra de test.
- El resto de las limitaciones (basadas en resultados de entrenamiento y evaluación) se
  completan en fase de modelado.

## Uso previsto

Proyecto académico — no apto para uso clínico real sin validación adicional.

Dentro de ese marco, el modelo solo es aplicable a **cortes 2D de resonancia T1 con
contraste**, la única secuencia representada en las clases con tumor del dataset de
entrenamiento. Sus predicciones sobre otras secuencias (T2, FLAIR) no son confiables.

---

**Ver también:** [documentación no técnica](documentacion_no_tecnica.md) ·
[documentación técnica](documentacion_tecnica.md) · [arquitectura en AWS](architecture.md) ·
[desarrollo asistido por IA](desarrollo_asistido_por_ia.md)
