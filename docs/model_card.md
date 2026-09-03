# Model Card

> Documento a completar durante la fase de modelado (Bimestre 1 - prototipado / Bimestre 2 - versión final en AWS).

## Descripción del modelo

- **Arquitectura:** TBD (EfficientNet-B0 / MobileNetV2)
- **Framework:** TBD (PyTorch / TensorFlow)
- **Técnica:** Transfer learning (backbone preentrenado en ImageNet + fine-tuning de capas superiores)
- **Resolución de entrada:** TBD

## Dataset

- **Fuente:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Clases:** Glioma, Meningioma, Pituitario, Sin tumor
- **Split:** TBD

## Métricas de evaluación

Priorizadas por contexto sanitario (un falso negativo es más costoso que un falso positivo):

| Métrica | Valor | Justificación |
|---|---|---|
| Recall / Sensibilidad | TBD | Minimizar falsos negativos (tumores no detectados) |
| F1-Score | TBD | Balance entre precisión y recall por clase |
| ROC-AUC | TBD | Capacidad de discriminación entre clases |
| Matriz de confusión | TBD | Detalle de errores por clase |

## Limitaciones

TBD

## Uso previsto

Proyecto académico — no apto para uso clínico real sin validación adicional.
