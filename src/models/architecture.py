"""
Definición de la arquitectura del modelo (transfer learning).

Candidatos: EfficientNet-B0 / MobileNetV2, con backbone preentrenado en
ImageNet y la mayoría de las capas congeladas (solo se entrena el head
de clasificación + últimas capas), para mantener el entrenamiento
viable en CPU dentro del free tier de SageMaker.

Decisión final y justificación: pendiente de fase de modelado (Bimestre 1).
"""


def build_model(num_classes: int = 4, architecture: str = "efficientnet_b0"):
    """Construye y devuelve el modelo con el backbone preentrenado + head de clasificación.

    TODO: implementar en fase de modelado (Bimestre 1), una vez definido
    el framework (PyTorch/TensorFlow) y la arquitectura base.
    """
    raise NotImplementedError
