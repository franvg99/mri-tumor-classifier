"""Tests unitarios para scripts/package_model.py"""

import subprocess
import sys
import tarfile

import pytest
import torch

from scripts.package_model import CODIGO, armar_paquete
from src.models.architecture import build_model

# Corre en un proceso aparte, sin acceso al repositorio, como en el contenedor.
PRUEBA_AISLADA = """
import io, json, sys
from pathlib import Path
raiz = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(raiz / "code"))
import inference, src
from PIL import Image
assert raiz in Path(src.__file__).resolve().parents, src.__file__
buffer = io.BytesIO()
Image.new("L", (64, 64), 120).save(buffer, format="JPEG")
modelo = inference.model_fn(str(raiz))
proba = inference.predict_fn(inference.input_fn(buffer.getvalue(), "image/jpeg"), modelo)
for pesado in ("sklearn", "pandas", "matplotlib"):
    assert pesado not in sys.modules, pesado
print(json.loads(inference.output_fn(proba, "application/json"))["clase"])
"""


@pytest.fixture(scope="module")
def paquete(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("paquete")
    torch.save(build_model(pretrained=False).state_dict(), carpeta / "model.pth")
    return armar_paquete(carpeta / "model.pth", carpeta / "model.tar.gz")


def test_contenido_del_paquete(paquete):
    with tarfile.open(paquete) as tar:
        assert set(tar.getnames()) == {"model.pth", *CODIGO}


def test_el_paquete_funciona_sin_el_repositorio(paquete, tmp_path):
    """Descomprimido en otra carpeta, responde sin depender de archivos que no incluye."""
    with tarfile.open(paquete) as tar:
        tar.extractall(tmp_path)
    proceso = subprocess.run(
        [sys.executable, "-I", "-c", PRUEBA_AISLADA, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert proceso.returncode == 0, proceso.stderr
    assert proceso.stdout.strip() in {"glioma", "meningioma", "pituitary", "notumor"}


def test_falla_si_no_hay_modelo_entrenado(tmp_path):
    with pytest.raises(FileNotFoundError, match="primero hay que entrenar"):
        armar_paquete(tmp_path / "no_existe.pth", tmp_path / "model.tar.gz")
