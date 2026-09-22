[⬅ Volver al README](../README.md)

# Model Card

> Documento a completar durante la fase de modelado (Bimestre 1 - prototipado / Bimestre 2 - versión final en AWS).
> Para el razonamiento detrás de cada decisión ya tomada, ver
> [`documentacion_tecnica.md`](documentacion_tecnica.md).

## Descripción del modelo

- **Arquitectura:** TBD (EfficientNet-B0 / MobileNetV2)
- **Framework:** TBD (PyTorch / TensorFlow)
- **Técnica:** Transfer learning (backbone preentrenado en ImageNet + fine-tuning de capas superiores)
- **Resolución de entrada:** 224×224 — elegida en el EDA (`notebooks/01_eda.ipynb`, sección 3) para alinear con la resolución preentrenada estándar de EfficientNet-B0/MobileNetV2 sobre ImageNet, aceptando upscaling en el 11.3% de las imágenes que llegan más chicas

## Dataset

- **Fuente:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Clases:** Glioma, Meningioma, Pituitario, Sin tumor
- **Split:** 80% train / 10% val / 10% test, estratificado por clase, seed fija (`config/config.yaml`). Deduplicado antes de dividir — el EDA encontró 4.72% de duplicados exactos.

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
  patología real. Mitigado parcialmente con augmentation de brillo/contraste, pero es una
  limitación estructural del dataset que no se puede eliminar del todo.
- El resto de las limitaciones (basadas en resultados de entrenamiento y evaluación) se
  completan en fase de modelado.

## Uso previsto

Proyecto académico — no apto para uso clínico real sin validación adicional.

---

**Ver también:** [documentación no técnica](documentacion_no_tecnica.md) ·
[documentación técnica](documentacion_tecnica.md) · [arquitectura en AWS](architecture.md) ·
[desarrollo asistido por IA](desarrollo_asistido_por_ia.md)
