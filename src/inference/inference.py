"""
Handlers de inferencia para el SageMaker Real-Time Endpoint.

SageMaker espera estas 4 funciones cuando se usa el modo "script mode"
para servir un modelo custom:
- model_fn: carga el modelo desde el path de artefactos
- input_fn: deserializa el request (imagen -> tensor)
- predict_fn: corre la inferencia
- output_fn: serializa la respuesta a JSON {clase, probabilidad}
"""


def model_fn(model_dir):
    """Carga el modelo entrenado desde model_dir.

    TODO: implementar en fase de despliegue (Bimestre 2).
    """
    raise NotImplementedError


def input_fn(request_body, request_content_type):
    """Deserializa el input (imagen) recibido por el endpoint.

    TODO: implementar en fase de despliegue (Bimestre 2).
    """
    raise NotImplementedError


def predict_fn(input_data, model):
    """Corre la inferencia sobre la imagen ya preprocesada.

    TODO: implementar en fase de despliegue (Bimestre 2).
    """
    raise NotImplementedError


def output_fn(prediction, response_content_type):
    """Serializa la predicción a JSON: {"clase": ..., "probabilidad": ...}.

    TODO: implementar en fase de despliegue (Bimestre 2).
    """
    raise NotImplementedError
