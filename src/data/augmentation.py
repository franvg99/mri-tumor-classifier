"""
Data augmentation para imágenes MRI cerebrales.

Transformaciones aplicadas SOLO sobre el set de entrenamiento:
- Rotaciones leves (+/-10-15 grados)
- Flip horizontal
- Zoom / crop leve
- Ajustes de brillo/contraste

Se evitan transformaciones agresivas (ej. deformaciones elásticas fuertes)
que podrían distorsionar features clínicamente relevantes.
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
