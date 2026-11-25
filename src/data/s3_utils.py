"""
Utilidades para subir/descargar datos de S3 usando boto3.

Se usa desde el entorno local con la sesión de `aws login` (el perfil se elige
con la variable de entorno `AWS_PROFILE`) y desde SageMaker con su rol de IAM.
En ningún caso se escriben credenciales en el código.
"""

from pathlib import Path

import boto3

REGION = "us-east-2"  # región asignada por AWS al proyecto; no se puede cambiar


def get_s3_client(region: str = REGION):
    """Crea un cliente de S3."""
    return boto3.client("s3", region_name=region)


def upload_file(local_path: str, bucket: str, s3_key: str, region: str = REGION):
    """Sube un archivo local a S3. Si `s3_key` ya existe, lo reemplaza."""
    get_s3_client(region).upload_file(str(local_path), bucket, s3_key)


def upload_directory(local_dir: str, bucket: str, s3_prefix: str, region: str = REGION):
    """Sube un directorio completo a S3, preservando la estructura de carpetas por clase.

    Returns:
        La cantidad de archivos subidos.
    """
    raiz = Path(local_dir)
    if not raiz.is_dir():
        raise NotADirectoryError(f"No existe la carpeta {str(local_dir)!r}")

    cliente = get_s3_client(region)  # un solo cliente para todos los archivos
    archivos = sorted(p for p in raiz.rglob("*") if p.is_file())
    for archivo in archivos:
        # as_posix(): S3 separa con "/" aunque Windows use "\".
        clave = f"{s3_prefix.strip('/')}/{archivo.relative_to(raiz).as_posix()}".lstrip("/")
        cliente.upload_file(str(archivo), bucket, clave)
    return len(archivos)


def download_file(bucket: str, s3_key: str, local_path: str, region: str = REGION):
    """Descarga un archivo de S3, creando la carpeta de destino si no existe."""
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    get_s3_client(region).download_file(bucket, s3_key, str(local_path))
