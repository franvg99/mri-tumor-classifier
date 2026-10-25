"""
Definición de la arquitectura del modelo (transfer learning).

EfficientNet-B0 preentrenada en ImageNet, con el backbone congelado y una
capa final nueva de 4 clases. Solo se entrena esa capa final (~5.000
parámetros sobre ~4 millones), lo que hace viable el entrenamiento en CPU
dentro del free tier de SageMaker (`ml.m5.xlarge`, sin GPU).

Justificación de la elección frente a MobileNetV2: con el backbone congelado
no hay backward pass a través de él, por lo que la diferencia de costo entre
ambas es irrelevante; en cambio, la mejor accuracy de transferencia de
EfficientNet-B0 pesa más, dado que la métrica prioritaria es el recall
(ver `docs/model_card.md`).

Estructura de EfficientNet-B0 en torchvision:

- `model.features`: backbone convolucional que extrae los patrones de la imagen.
- `model.avgpool`: resume cada mapa de features en un vector de 1280 valores.
- `model.classifier`: `Sequential(Dropout(0.2), Linear(1280, 1000))`. Acá se
  reemplaza la `Linear` por una de 4 salidas.
"""

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ARQUITECTURAS_SOPORTADAS = ("efficientnet_b0",)


def build_model(
    num_classes: int = 4,
    architecture: str = "efficientnet_b0",
    pretrained: bool = True,
) -> nn.Module:
    """Construye EfficientNet-B0 con el backbone congelado y un head de `num_classes`.

    El modelo devuelve logits (puntajes sin normalizar), no probabilidades: la
    función de pérdida `CrossEntropyLoss` aplica el softmax internamente, y
    agregarlo acá lo aplicaría dos veces.

    Args:
        num_classes: cantidad de clases de salida.
        architecture: arquitectura base. Solo se admite `"efficientnet_b0"`.
        pretrained: si es `True`, carga los pesos de ImageNet (la primera vez
            los descarga, ~20 MB). Los tests usan `False` para no depender de
            la descarga; sin pesos preentrenados, congelar el backbone no tiene
            sentido fuera de esos tests.

    Returns:
        El modelo, con `requires_grad=False` en todo `model.features` y
        `requires_grad=True` solo en el head de clasificación.

    Raises:
        ValueError: si `architecture` no es una arquitectura soportada.
    """
    if architecture not in ARQUITECTURAS_SOPORTADAS:
        raise ValueError(
            f"Arquitectura no soportada: {architecture!r}. "
            f"Opciones: {ARQUITECTURAS_SOPORTADAS}"
        )

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Congelar el backbone: sus pesos quedan fijos durante el entrenamiento.
    for param in model.features.parameters():
        param.requires_grad = False

    # Reemplazar la capa final (1280 -> 1000 de ImageNet) por una de 4 clases.
    # Una capa nueva se crea con requires_grad=True, así que es lo único que
    # se entrena.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
