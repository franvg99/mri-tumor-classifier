[⬅ Volver al README](../README.md)

# Model Card

> Documento a completar durante la fase de modelado (Bimestre 1 - prototipado / Bimestre 2 - versión final en AWS).
> Para el razonamiento detrás de cada decisión ya tomada, ver
> [`documentacion_tecnica.md`](documentacion_tecnica.md).

## Descripción del modelo

- **Arquitectura:** EfficientNet-B0 (backbone preentrenado en ImageNet, congelado)
- **Framework:** PyTorch
- **Técnica:** Transfer learning con extracción de features: el backbone preentrenado en ImageNet queda congelado y solo se entrena la capa final de 4 clases (5.124 parámetros entrenables sobre 4.012.672 totales)
- **Resolución de entrada:** 224×224 — elegida en el EDA (`notebooks/01_eda.ipynb`, sección 3) para alinear con la resolución preentrenada estándar de EfficientNet-B0/MobileNetV2 sobre ImageNet, aceptando upscaling en el 11.3% de las imágenes que llegan más chicas

## Dataset

- **Fuente:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Clases:** Glioma, Meningioma, Pituitario, Sin tumor
- **Split:** 70% train / 10% val / 20% test, estratificado por clase, seed fija (`config/config.yaml`). Deduplicado antes de dividir — el EDA encontró 4.72% de duplicados exactos. El test es más grande que el val para acotar el margen de error del recall por clase (~350 imágenes por clase en lugar de ~175); con el backbone congelado, la reducción del train tiene un costo bajo.

## Entrenamiento

- **Optimizador:** Adam, `learning_rate = 0.001`, batch de 32 imágenes.
- **Función de pérdida:** `CrossEntropyLoss`.
- **Criterio de parada:** early stopping sobre el recall promedio (macro) de validación,
  con un techo de 30 épocas y una paciencia de 5. Se conserva el modelo de la mejor época,
  no el de la última. Se eligió el recall en lugar de la loss por ser la métrica
  prioritaria del proyecto.
- **Corrida de referencia (25/10/2026, entorno local, CPU):** cortó en la época 20; la
  mejor fue la 15. Duración total: 52 minutos (~2,6 minutos por época).

Resultados en **validación** (701 imágenes) de la época 15. No reemplazan a las métricas
de test de la sección siguiente: la validación se usó para elegir la época, por lo que su
resultado es levemente optimista.

| Clase | Recall en validación |
|---|---|
| Glioma | 83,2% |
| Meningioma | 91,0% |
| Pituitario | 98,9% |
| Sin tumor | 98,2% |
| **Promedio (macro)** | **92,8%** |

Imágenes con tumor clasificadas como "Sin tumor" (el falso negativo clínicamente más
grave): 3 de 533 (0,6%).

Observaciones de las curvas:

- La loss de validación desciende junto con la de train y no vuelve a subir: no hay
  evidencia de sobreajuste. Una comparación sobre la época 1 dio el mismo recall promedio
  en train (sin augmentation) y en validación (87%).
- El recall de glioma y el de meningioma oscilan en sentidos opuestos entre épocas (en la
  época 16, glioma sube a 94,4% y meningioma baja a 70,8%). Sugiere que el modelo confunde
  estas dos clases entre sí; se confirmará con la matriz de confusión de test.
- Desde la época 5 el recall promedio se mantiene entre 91% y 93%. La diferencia entre
  las mejores épocas está dentro de la variación esperable para el tamaño de validación.

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
