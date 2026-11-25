[⬅ Volver al README](../README.md)

# Model Card

> Ficha técnica del modelo entrenado el 25/10/2026 y desplegado en AWS el 25/11/2026.
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
- **Split:** 70% train / 10% val / 20% test, estratificado por clase, seed fija (`config/config.yaml`). Antes de dividir se eliminaron los duplicados exactos (archivos idénticos, 4.72% del dataset). Ese filtro no detecta copias casi idénticas, que quedaron repartidas entre los tres conjuntos (ver [Limitaciones](#limitaciones)). El test es más grande que el val para acotar el margen de error del recall por clase (~350 imágenes por clase en lugar de ~175); con el backbone congelado, la reducción del train tiene un costo bajo.

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
resultado es levemente optimista. Además, 118 de las 701 (16,8%) tienen una copia casi
idéntica en train (ver Limitaciones).

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

- La loss de validación desciende junto con la de train y termina cerca de ella: no hay
  señales claras de sobreajuste. Sube levemente entre las épocas 17 y 20 (de 0,201 a
  0,224), un movimiento similar a las oscilaciones anteriores. Una comparación sobre la época 1 dio el mismo recall promedio
  en train (sin augmentation) y en validación (87%).
- El recall de glioma y el de meningioma oscilan en sentidos opuestos entre épocas (en la
  época 16, glioma sube a 94,4% y meningioma baja a 70,8%). Sugiere que el modelo confunde
  estas dos clases entre sí; se confirmará con la matriz de confusión de test.
- Desde la época 5 el recall promedio se mantiene entre 90,7% y 92,8%. La diferencia entre
  las mejores épocas está dentro de la variación esperable para el tamaño de validación.

## Métricas de evaluación

Priorizadas por contexto sanitario (un falso negativo es más costoso que un falso positivo).
Evaluación del 25/11/2026 sobre **1.146 imágenes de test**, reproducible con
`python -m src.models.evaluate --data-dir <dataset>`.

> **Corrección del 25/11/2026.** La evaluación inicial (25/10/2026) utilizó las 1.403
> imágenes de test y dio un recall promedio de 93,0%. Al analizar los errores detecté que
> 257 de ellas (18,3%) tienen una copia casi idéntica en train, que el deduplicado por
> archivo no detecta. Acertarlas no demuestra que el modelo generalice, por lo que las
> excluí: el resultado corregido es **92,5%**. El modelo es el mismo. Ver
> [Efecto de las copias](#efecto-de-las-copias-entre-train-y-test).

| Métrica (macro) | Valor | Justificación |
|---|---|---|
| Recall / Sensibilidad | 92,5% | Minimizar falsos negativos (tumores no detectados) |
| Precisión | 91,5% | Proporción de aciertos entre las predicciones de cada clase |
| F1-Score | 91,9% | Balance entre precisión y recall por clase |
| ROC-AUC (one-vs-rest) | 0,987 | Capacidad de discriminación entre clases |
| Accuracy | 91,7% | Referencia; no es la métrica prioritaria |

Por clase (margen de error aproximado del recall: entre ±2 y ±3,5 puntos):

| Clase | Recall | Precisión | F1 | ROC-AUC | Imágenes |
|---|---|---|---|---|---|
| Glioma | 86,7% | 96,8% | 91,5% | 0,984 | 354 |
| Meningioma | 91,4% | 83,7% | 87,4% | 0,974 | 325 |
| Pituitario | 95,5% | 96,0% | 95,7% | 0,997 | 352 |
| Sin tumor | 96,5% | 89,5% | 92,9% | 0,994 | 115 |

La precisión, el F1 y la accuracy dependen de la proporción de cada clase. Como la exclusión
afecta sobre todo a "Sin tumor", no son directamente comparables con los de la evaluación
inicial; el recall por clase sí lo es.

Matriz de confusión (filas: clase real; columnas: predicción):

| | Glioma | Meningioma | Pituitario | Sin tumor |
|---|---|---|---|---|
| **Glioma** | 307 | 43 | 1 | 3 |
| **Meningioma** | 7 | 297 | 12 | 9 |
| **Pituitario** | 2 | 13 | 336 | 1 |
| **Sin tumor** | 1 | 2 | 1 | 111 |

Lectura de los resultados:

- **El recall de test (92,5%) es similar al de validación (92,8%).** La validación también
  contiene copias de train (ver Limitaciones), por lo que la comparación es orientativa.
- **Confusión principal: glioma predicho como meningioma** (43 casos, 12% de los
  gliomas), con una asimetría marcada (solo 7 meningiomas predichos como glioma).
  Explica el recall más bajo de glioma y la precisión más baja de meningioma. En estos
  casos el tumor se detecta, aunque se clasifica mal su tipo.
- **Falsos negativos graves (tumor predicho como "Sin tumor"): 13 de 1.031 (1,3%).**
  9 de los 13 son meningiomas, y son 10 imágenes distintas (tres aparecen repetidas). No se
  reparten al azar: ver [Errores según el tamaño](#errores-según-el-tamaño-de-la-imagen).
- **"Sin tumor" se mide sobre solo 115 imágenes**, por lo que su recall tiene un margen de
  unos ±3,3 puntos.

### Efecto de las copias entre train y test

Considero copia a una imagen de test cuya correlación con alguna de train, en escala de
grises y a 32×32 píxeles, es de 0,999 o más (`src/data/similitud.py`).

| | Evaluación inicial | Sin copias |
|---|---|---|
| Imágenes | 1.403 | 1.146 |
| Recall promedio | 93,0% | **92,5%** |
| Recall glioma | 86,6% | 86,7% |
| Recall meningioma | 91,6% | 91,4% |
| Recall pituitario | 95,2% | 95,5% |
| Recall sin tumor | 98,8% | **96,5%** |
| Tumores predichos como "Sin tumor" | 14 de 1.067 | 13 de 1.031 |

- **Se concentra en "Sin tumor":** 221 de las 257 copias. El modelo acierta el 100% de esas
  221, contra el 96,5% de las que no tienen copia.
- **El recall de las clases con tumor casi no varía.**
- **El umbral es una elección.** Se fijó conociendo los resultados de los cuatro valores, por
  lo que informo la sensibilidad completa: los umbrales más estrictos bajan el promedio hasta
  91,8%.

| Umbral | Imágenes | Recall promedio | Recall sin tumor | Tumores predichos como "Sin tumor" |
|---|---|---|---|---|
| 0,98 | 997 | 91,8% | 94,7% (57 imágenes) | 13 de 940 |
| 0,99 | 1.099 | 92,3% | 96,1% (76) | 13 de 1.023 |
| 0,995 | 1.128 | 92,4% | 96,0% (99) | 13 de 1.029 |
| **0,999** | **1.146** | **92,5%** | **96,5% (115)** | **13 de 1.031** |
| Sin excluir | 1.403 | 93,0% | 98,8% (336) | 14 de 1.067 |

### Errores según el tamaño de la imagen

El modelo no conoce el tamaño original, porque toda imagen se lleva a 224×224. En este
dataset, sin embargo, el tamaño acompaña a la fuente de origen: entre el 93% y el 98% de las
imágenes con tumor de test mide exactamente 512 px, y ninguna imagen sin tumor lo hace.

| Imágenes de test sin copia | Imágenes | Predichas como "Sin tumor" |
|---|---|---|
| Con tumor, 300 px o más | 1.008 | 3 (0,3%) |
| Con tumor, menos de 300 px | 23 | **10 (43,5%)** |
| Sin tumor, 300 px o más | 32 | 31 (96,9%) |
| Sin tumor, menos de 300 px | 83 | 80 (96,4%) |

- **El 1,3% general oculta dos situaciones muy distintas.** Vale sobre todo para imágenes como
  las de la fuente principal, y no puede trasladarse a otras.
- **Es compatible con el atajo de la fuente** descrito en Limitaciones: los tumores cuyo
  aspecto se parece al de las imágenes sin tumor se confunden con ellas.
- **Alcance:** es una asociación y no prueba la causa, porque el tamaño, la fuente y la
  secuencia varían juntos. Con 23 casos (algunos repetidos), el 43,5% es muy impreciso:
  aproximadamente entre 26% y 63%. El modelo no responde solo al tamaño, ya que reconoce 31 de
  las 32 imágenes sin tumor de 300 px o más.

Análisis completo en `notebooks/03_model_experiments.ipynb`, sección 4.

### Prueba de estrés de brillo

Se multiplicó el brillo de las 1.146 imágenes por distintos factores para verificar si el
modelo usa el brillo como atajo (ver Limitaciones). El augmentation de train ya varía el
brillo entre 0,6 y 1,4, por lo que el rango se extendió hasta 1,8, la relación de brillo
medio entre "Sin tumor" y glioma medida en el EDA.

| Factor de brillo | Tumores predichos como "Sin tumor" | "Sin tumor" reconocidos |
|---|---|---|
| 0,50 | 1,3% | 92,2% |
| 0,75 | 1,4% | 95,7% |
| 1,00 (original) | 1,3% | 96,5% |
| 1,25 | 1,6% | 95,7% |
| 1,50 | 2,2% | 96,5% |
| 1,80 | 3,2% | 96,5% |

- **El brillo no es la señal principal del modelo.** Oscurecer a la mitad las imágenes
  sin tumor reduce su reconocimiento de 96,5% a 92,2% (5 imágenes de 115), y aclarar las
  imágenes con tumor no provoca un desplazamiento masivo hacia "Sin tumor".
- **Existe una sensibilidad leve, que crece fuera del rango del augmentation.** Con un
  factor de 1,8, los falsos negativos graves pasan de 1,3% a 3,2% (de 13 a 33 imágenes).
  Parte de ese aumento puede deberse a la saturación de píxeles en blanco, que degrada la
  imagen, y no exclusivamente al atajo.
- **Alcance de la prueba:** solo evalúa el brillo global. No descarta que el modelo use
  otras características propias de la fuente, en particular el tipo de secuencia de
  resonancia (ver Limitaciones), que requiere un etiquetado manual.

## Limitaciones

- **Copias casi idénticas repartidas entre train, validación y test.** El deduplicado
  compara archivos byte a byte, y no detecta una misma imagen guardada de nuevo o con otro
  contraste o nitidez. La evaluación de test las excluye, pero el modelo se entrenó y su
  época se eligió con ellas presentes (118 de las 701 imágenes de validación). Además, las
  imágenes "Sin tumor" realmente distintas son muchas menos que las disponibles: unas 863
  de 1.681. La corrección completa es rehacer la división y reentrenar, prevista en
  SageMaker si AWS aprueba la cuota solicitada antes del 02/12/2026.
- **Sin identificación de pacientes.** El dataset no indica a qué paciente corresponde cada
  imagen, por lo que cortes distintos de una misma persona pueden estar en train y en test.
  Excluir las copias no resuelve este punto.
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
  La prueba de estrés de brillo (25/10/2026) mostró que el brillo global no es la señal
  principal del modelo, aunque existe una sensibilidad leve fuera del rango del
  augmentation (ver Métricas de evaluación).
- **Confusión entre glioma y meningioma:** el 12% de los gliomas de test se clasifica
  como meningioma. El tumor se detecta, pero el tipo informado no es confiable en esos
  casos.
- **Falsos negativos graves concentrados en un tipo de imagen:** 10 de los 13 tumores de test
  predichos como "Sin tumor" miden menos de 300 px, cuando solo 23 de los 1.031 tumores
  tienen ese tamaño. 9 de los 13 son meningiomas. La tasa general de 1,3% no es aplicable a
  imágenes de otras fuentes.

## Despliegue

El modelo evaluado en esta ficha es el mismo que se publicó en un endpoint de SageMaker el
25/11/2026 (`ml.m5.xlarge`, CPU, PyTorch 2.6). El código de inferencia reproduce la matriz
de confusión de test en local, y el endpoint dio la misma clase que el modelo local en 40
de 40 imágenes de test. El endpoint se eliminó al terminar la prueba. Detalle en
[arquitectura](architecture.md).

## Uso previsto

Proyecto académico — no apto para uso clínico real sin validación adicional.

Dentro de ese marco, el modelo solo es aplicable a **cortes 2D de resonancia T1 con
contraste**, la única secuencia representada en las clases con tumor del dataset de
entrenamiento. Sus predicciones sobre otras secuencias (T2, FLAIR) no son confiables.

---

**Ver también:** [documentación no técnica](documentacion_no_tecnica.md) ·
[documentación técnica](documentacion_tecnica.md) · [arquitectura en AWS](architecture.md) ·
[desarrollo asistido por IA](desarrollo_asistido_por_ia.md)
