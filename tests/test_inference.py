"""Tests unitarios para src/inference/inference.py"""

import io
import json

import pytest
import torch
from PIL import Image

from src.data.dataset import CLASES
from src.inference.inference import input_fn, model_fn, output_fn, predict_fn
from src.models.architecture import build_model


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """Carpeta con un `model.pth` sin entrenar, como la que arma SageMaker."""
    carpeta = tmp_path_factory.mktemp("modelo")
    torch.save(build_model(pretrained=False).state_dict(), carpeta / "model.pth")
    return str(carpeta)


def imagen_en_bytes(formato="JPEG", modo="L"):
    buffer = io.BytesIO()
    Image.new(modo, (300, 200), color=120).save(buffer, format=formato)
    return buffer.getvalue()


def test_model_fn_carga_los_pesos_en_modo_evaluacion(model_dir):
    model = model_fn(model_dir)
    guardados = torch.load(f"{model_dir}/model.pth", weights_only=True)

    assert not model.training
    assert all(torch.equal(v, guardados[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize(
    "formato, content_type", [("JPEG", "image/jpeg"), ("PNG", "image/png")]
)
def test_input_fn_devuelve_un_batch_de_una_imagen(formato, content_type):
    """Una imagen en escala de grises de cualquier tamaño sale con la forma del modelo."""
    assert input_fn(imagen_en_bytes(formato), content_type).shape == (1, 3, 224, 224)


def test_input_fn_rechaza_formato_no_admitido():
    with pytest.raises(ValueError, match="Formato no admitido"):
        input_fn(b"a,b,c", "text/csv")


def test_input_fn_rechaza_contenido_que_no_es_imagen():
    with pytest.raises(ValueError, match="no es una imagen válida"):
        input_fn(b"esto no es una imagen", "image/jpeg")


def test_input_fn_rechaza_imagen_truncada():
    contenido = imagen_en_bytes()
    with pytest.raises(ValueError, match="no es una imagen válida"):
        input_fn(contenido[: len(contenido) // 2], "image/jpeg")


def test_output_fn_informa_la_clase_ganadora_y_las_4_probabilidades():
    respuesta = json.loads(output_fn(torch.tensor([[0.1, 0.6, 0.2, 0.1]]), "application/json"))
    assert respuesta == {
        "clase": "meningioma",
        "probabilidad": pytest.approx(0.6),
        "probabilidades": pytest.approx(
            {"glioma": 0.1, "meningioma": 0.6, "pituitary": 0.2, "notumor": 0.1}
        ),
    }


def test_output_fn_rechaza_formato_no_admitido():
    with pytest.raises(ValueError, match="respuesta no admitido"):
        output_fn(torch.tensor([[0.1, 0.6, 0.2, 0.1]]), "text/csv")


def test_recorrido_completo(model_dir):
    """Las 4 funciones encadenadas, en el orden en que las llama SageMaker."""
    model = model_fn(model_dir)
    proba = predict_fn(input_fn(imagen_en_bytes(), "image/jpeg"), model)
    respuesta = json.loads(output_fn(proba, "application/json"))

    assert proba.sum().item() == pytest.approx(1)
    assert respuesta["clase"] in CLASES
    assert respuesta["probabilidad"] == pytest.approx(proba.max().item())
