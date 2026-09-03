"""
Cliente simple para invocar el endpoint de SageMaker ya desplegado
(útil para probarlo manualmente o desde un notebook).
"""


def predict_image(image_path: str, endpoint_name: str, region: str = "us-east-1"):
    """Envía una imagen al endpoint y devuelve la respuesta JSON {clase, probabilidad}.

    TODO: implementar en fase de despliegue (Bimestre 2).
    """
    raise NotImplementedError
