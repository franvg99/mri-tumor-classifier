"""
Handlers de inferencia para el SageMaker Real-Time Endpoint.

SageMaker espera estas 4 funciones cuando se usa el modo "script mode"
para servir un modelo custom (los nombres y las firmas son fijos):

- model_fn: carga el modelo desde el path de artefactos (una vez, al iniciar)
- input_fn: deserializa el request (imagen -> tensor)
- predict_fn: corre la inferencia
- output_fn: serializa la respuesta a JSON {clase, probabilidad, probabilidades}

El preprocesamiento no se define acá: se importa `get_eval_pipeline()`, para que
el endpoint prepare cada imagen exactamente igual que la evaluación sobre test.

Este módulo corre dentro del contenedor de SageMaker. Por eso no importa
`evaluate.py`, que arrastra dependencias innecesarias para servir el modelo
(scikit-learn, pandas, matplotlib).
"""

import io
import json
import os

import torch
from PIL import Image, UnidentifiedImageError

from src.data.augmentation import get_eval_pipeline
from src.data.dataset import CLASES
from src.models.architecture import build_model

TIPOS_DE_IMAGEN = ("image/jpeg", "image/png")
TIPO_RESPUESTA = "application/json"


def model_fn(model_dir):
    """Carga el modelo entrenado (`model.pth`) desde `model_dir`.

    Usa `pretrained=False` porque los pesos, incluidos los del backbone, ya están
    en `model.pth`: el contenedor no necesita descargar los de ImageNet.
    """
    model = build_model(num_classes=len(CLASES), pretrained=False)
    pesos = torch.load(
        os.path.join(model_dir, "model.pth"), map_location="cpu", weights_only=True
    )
    model.load_state_dict(pesos)
    model.eval()  # desactiva el Dropout: la misma imagen da siempre la misma respuesta
    return model


def input_fn(request_body, request_content_type):
    """Deserializa la imagen recibida y la prepara para el modelo.

    Args:
        request_body: contenido del archivo de imagen, en bytes.
        request_content_type: formato declarado por el cliente. Se admiten los
            de `TIPOS_DE_IMAGEN`.

    Returns:
        Tensor `(1, 3, 224, 224)`.

    Raises:
        ValueError: si el formato no está admitido o el contenido no es una
            imagen válida.
    """
    if request_content_type not in TIPOS_DE_IMAGEN:
        raise ValueError(
            f"Formato no admitido: {request_content_type!r}. Opciones: {TIPOS_DE_IMAGEN}"
        )

    try:
        # Misma conversión que MRIDataset: el canal gris se replica en los 3.
        imagen = Image.open(io.BytesIO(request_body)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("El contenido recibido no es una imagen válida") from error

    return get_eval_pipeline()(imagen).unsqueeze(0)  # agrega la dimensión del batch


def predict_fn(input_data, model):
    """Corre la inferencia sobre la imagen ya preprocesada.

    Returns:
        Tensor `(1, 4)` con la probabilidad de cada clase, en el orden de `CLASES`.
    """
    with torch.no_grad():  # sin gradientes: acá no se entrena
        return torch.softmax(model(input_data), dim=1)


def output_fn(prediction, response_content_type):
    """Serializa la predicción a JSON.

    Además de la clase ganadora y su probabilidad, informa las 4 probabilidades:
    permite ver la segunda opción del modelo, relevante porque su confusión
    principal es entre glioma y meningioma (ver docs/model_card.md).

    Returns:
        `{"clase": ..., "probabilidad": ..., "probabilidades": {clase: ...}}`

    Raises:
        ValueError: si el cliente pide un formato distinto de JSON.
    """
    if response_content_type not in (TIPO_RESPUESTA, "*/*", None):
        raise ValueError(
            f"Formato de respuesta no admitido: {response_content_type!r}. "
            f"Solo se admite {TIPO_RESPUESTA!r}"
        )

    probabilidades = dict(zip(CLASES, prediction[0].tolist()))
    clase = max(probabilidades, key=probabilidades.get)
    return json.dumps(
        {"clase": clase, "probabilidad": probabilidades[clase], "probabilidades": probabilidades}
    )
