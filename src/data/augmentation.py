"""
Data augmentation para imágenes MRI cerebrales.

Transformaciones aplicadas SOLO sobre el set de entrenamiento:
- Rotaciones leves (+/-10-15 grados)
- Zoom / crop leve
- Ajustes de brillo/contraste (obligatorio, ver justificación abajo)
- SIN flip horizontal (excluido a propósito, ver justificación abajo)

Se evitan transformaciones agresivas (ej. deformaciones elásticas fuertes)
que podrían distorsionar features clínicamente relevantes.

Justificación de las dos decisiones no obvias (EDA, notebook 01, secciones 4 y 5):

- **Sin flip horizontal:** el dataset mezcla cortes axial/sagital/coronal sin
  etiquetar. Un flip horizontal es válido en axial/coronal pero anatómicamente
  inválido en sagital (invierte adelante-atrás). Sin forma barata de detectar
  la orientación por imagen, se descarta el flip antes que arriesgar corromper
  el subconjunto sagital.
- **Brillo/contraste obligatorio:** hay diferencias sistemáticas de intensidad
  entre clases que coinciden con que el dataset combina 3 fuentes distintas
  (glioma=figshare, meningioma/pituitario≈SARTAJ, notumor=100% Br35H). Sin este
  augmentation, el modelo tiene un atajo disponible para aprender "de qué
  fuente vino la imagen" en vez de la patología real.
"""


def get_train_augmentation_pipeline():
    """Devuelve el pipeline de augmentation para entrenamiento.

    TODO: implementar en fase de modelado (Bimestre 1), usando la librería
    del framework elegido (torchvision.transforms / tf.keras.layers).
    """
    raise NotImplementedError


def get_eval_pipeline():
    """Pipeline sin augmentation, para validation/test (solo resize + normalize).

    TODO: implementar en fase de modelado (Bimestre 1).
    """
    raise NotImplementedError
