"""
Cliente simple para invocar el endpoint de SageMaker ya desplegado
(útil para probarlo manualmente o desde un notebook).

Uso (desde la raíz del proyecto, con el endpoint activo):
    python -m src.inference.predictor <imagen.jpg>
"""

import argparse
import json
import sys
from pathlib import Path

import boto3

REGION = "us-east-2"
CONTENT_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def predict_image(image_path: str, endpoint_name: str, region: str = REGION):
    """Envía una imagen al endpoint y devuelve la respuesta JSON
    {clase, probabilidad, probabilidades}.

    Raises:
        ValueError: si la extensión del archivo no es JPG ni PNG.
    """
    ruta = Path(image_path)
    content_type = CONTENT_TYPES.get(ruta.suffix.lower())
    if content_type is None:
        raise ValueError(f"Extensión no admitida: {ruta.suffix!r}. Opciones: {sorted(CONTENT_TYPES)}")

    respuesta = boto3.client("sagemaker-runtime", region_name=region).invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType=content_type,
        Accept="application/json",
        Body=ruta.read_bytes(),
    )
    return json.loads(respuesta["Body"].read())


def main(argv=None) -> int:
    """Punto de entrada: muestra la predicción de una imagen."""
    parser = argparse.ArgumentParser(description="Predicción con el endpoint de SageMaker")
    parser.add_argument("imagen")
    parser.add_argument("--endpoint-name", default="mri-tumor-classifier")
    args = parser.parse_args(argv)

    print(json.dumps(predict_image(args.imagen, args.endpoint_name), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
