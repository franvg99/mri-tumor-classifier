"""
Preprocesamiento de imágenes MRI: resize y normalización.

Prototipado en Google Colab (Bimestre 1); este módulo se porta luego
al Studio Notebook de SageMaker sin modificaciones (mismo código,
distinto entorno de ejecución).
"""


def resize_image(image, target_size: int = 150):
    """Redimensiona una imagen al tamaño de entrada del modelo.

    TODO: implementar en fase de modelado (Bimestre 1).
    """
    raise NotImplementedError


def normalize_image(image):
    """Normaliza los valores de píxel (0-255 -> rango esperado por el backbone preentrenado).

    TODO: implementar en fase de modelado (Bimestre 1).
    """
    raise NotImplementedError


def split_dataset(image_paths, train_split=0.8, val_split=0.1, test_split=0.1, seed=42):
    """Divide el dataset en train/val/test de forma estratificada por clase.

    TODO: implementar en fase de modelado (Bimestre 1).
    """
    raise NotImplementedError
