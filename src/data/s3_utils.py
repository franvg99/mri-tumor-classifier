"""
Utilidades para subir/descargar datos de S3 usando boto3.

Se usa desde Colab en el Bimestre 1 (con credenciales de AWS configuradas
localmente) y desde el Studio Notebook en el Bimestre 2 (con el rol IAM
del notebook).
"""

import boto3


def get_s3_client(region: str = "us-east-1"):
    """Crea un cliente de S3."""
    return boto3.client("s3", region_name=region)


def upload_file(local_path: str, bucket: str, s3_key: str, region: str = "us-east-1"):
    """Sube un archivo local a S3.

    TODO: implementar en fase de ingesta de datos (Bimestre 1).
    """
    raise NotImplementedError


def upload_directory(local_dir: str, bucket: str, s3_prefix: str, region: str = "us-east-1"):
    """Sube un directorio completo a S3, preservando la estructura de carpetas por clase.

    TODO: implementar en fase de ingesta de datos (Bimestre 1).
    """
    raise NotImplementedError


def download_file(bucket: str, s3_key: str, local_path: str, region: str = "us-east-1"):
    """Descarga un archivo de S3.

    TODO: implementar en fase de ingesta de datos (Bimestre 1).
    """
    raise NotImplementedError
