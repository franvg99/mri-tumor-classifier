"""
Arma el paquete `model.tar.gz` que SageMaker necesita para publicar el modelo y,
opcionalmente, lo sube a S3.

Estructura del paquete (la exige el contenedor de PyTorch de SageMaker):

    model.pth             pesos entrenados
    code/inference.py     handlers (copia de src/inference/inference.py)
    code/src/...          módulos de src/ que inference.py importa

Solo se incluyen los módulos necesarios para inferir: los de entrenamiento y
evaluación agregarían dependencias que el contenedor no necesita.

Uso (desde la raíz del proyecto):
    python -m scripts.package_model
    python -m scripts.package_model --bucket <bucket>
"""

import argparse
import sys
import tarfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# Ruta dentro del paquete -> archivo del repositorio.
CODIGO = {
    "code/inference.py": "src/inference/inference.py",
    "code/src/__init__.py": "src/__init__.py",
    "code/src/data/__init__.py": "src/data/__init__.py",
    "code/src/data/augmentation.py": "src/data/augmentation.py",
    "code/src/data/dataset.py": "src/data/dataset.py",
    "code/src/models/__init__.py": "src/models/__init__.py",
    "code/src/models/architecture.py": "src/models/architecture.py",
}


def _permisos(info):
    """Fija permisos de lectura: los de Windows no tienen equivalente en Linux."""
    info.mode = 0o644
    return info


def armar_paquete(model_path, destino):
    """Crea el `model.tar.gz` con los pesos y el código de inferencia.

    Returns:
        La ruta del paquete creado.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"No existe {model_path}: primero hay que entrenar el modelo")

    with tarfile.open(destino, "w:gz") as paquete:
        paquete.add(model_path, arcname="model.pth", filter=_permisos)
        for ruta_paquete, ruta_repo in CODIGO.items():
            paquete.add(RAIZ / ruta_repo, arcname=ruta_paquete, filter=_permisos)
    return Path(destino)


def main(argv=None) -> int:
    """Punto de entrada: arma el paquete y, si se indica un bucket, lo sube."""
    parser = argparse.ArgumentParser(description="Empaquetado del modelo para SageMaker")
    parser.add_argument("--model-path", default="models/model.pth")
    parser.add_argument("--output", default="models/model.tar.gz")
    parser.add_argument("--bucket", help="Si se indica, sube el paquete a este bucket de S3.")
    parser.add_argument("--s3-key", default="models/model.tar.gz")
    args = parser.parse_args(argv)

    paquete = armar_paquete(args.model_path, args.output)
    print(f"Paquete: {paquete} ({paquete.stat().st_size / 1e6:.1f} MB)")

    if args.bucket:
        from src.data.s3_utils import upload_file

        upload_file(paquete, args.bucket, args.s3_key)
        print(f"Subido a s3://{args.bucket}/{args.s3_key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
