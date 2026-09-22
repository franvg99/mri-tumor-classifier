"""
Preprocesamiento de imágenes MRI: resize y normalización.

Prototipado local (Bimestre 1); este módulo se porta luego al Studio
Notebook de SageMaker sin modificaciones (mismo código, distinto
entorno de ejecución).
"""


def resize_image(image, target_size: int = 224):
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

    Ignora el split Training/Testing original de Kaggle (`split_origen` en el DataFrame
    del EDA) — es un split ajeno, no el del proyecto.

    IMPORTANTE (hallazgo del EDA, sección 6): el dataset tiene 340 duplicados exactos
    (4.72%, 153 grupos, ninguno cruza clases). Hay que deduplicar ANTES de armar el
    split, quedándose con una sola copia por grupo de hash — si no, una imagen duplicada
    puede terminar con una copia en train y la otra en test, inflando las métricas de
    test. Usar `eda.detectar_duplicados()` como referencia de la lógica de detección.

    TODO: implementar en fase de modelado (Bimestre 1).
    """
    raise NotImplementedError
