"""Tests unitarios para src/models/architecture.py y evaluate.py"""

import numpy as np
import pytest
import torch

from src.models.architecture import build_model
from src.models.evaluate import compute_metrics


@pytest.fixture(scope="module")
def modelo():
    """Modelo sin pesos preentrenados: los tests no dependen de la descarga."""
    return build_model(pretrained=False)


def test_build_model_output_shape(modelo):
    """Un batch de N imágenes devuelve N vectores de 4 logits."""
    modelo.eval()
    with torch.no_grad():
        salida = modelo(torch.randn(2, 3, 224, 224))
    assert salida.shape == (2, 4)


def test_build_model_backbone_congelado(modelo):
    """Solo el head de clasificación es entrenable; el backbone queda fijo."""
    assert all(not p.requires_grad for p in modelo.features.parameters())
    assert all(p.requires_grad for p in modelo.classifier.parameters())

    entrenables = {n for n, p in modelo.named_parameters() if p.requires_grad}
    assert entrenables == {"classifier.1.weight", "classifier.1.bias"}


def test_build_model_arquitectura_invalida():
    with pytest.raises(ValueError, match="no soportada"):
        build_model(architecture="mobilenet_v2", pretrained=False)


def test_compute_metrics_ejemplo_a_mano():
    """Ejemplo chico con resultado calculable a mano.

    Índices: 0=glioma, 1=meningioma, 2=pituitary, 3=notumor.
    - glioma: 2 casos, uno acertado y uno predicho como notumor (falso negativo grave).
    - meningioma: 2 casos, uno acertado y uno confundido con glioma.
    - pituitary y notumor: 2 casos cada una, todos acertados.
    """
    y_true = [0, 0, 1, 1, 2, 2, 3, 3]
    y_pred = [0, 3, 1, 0, 2, 2, 3, 3]
    m = compute_metrics(y_true, y_pred)

    assert m["por_clase"]["glioma"]["recall"] == 0.5
    assert m["por_clase"]["glioma"]["precision"] == 0.5  # dijo glioma 2 veces, acertó 1
    assert m["por_clase"]["meningioma"]["recall"] == 0.5
    assert m["por_clase"]["meningioma"]["precision"] == 1.0
    assert m["por_clase"]["notumor"]["precision"] == pytest.approx(2 / 3)
    assert m["recall_macro"] == 0.75
    assert m["tumores_como_sanos"] == 1
    assert m["tumores_total"] == 6
    assert m["matriz_confusion"][0] == [1, 0, 0, 1]  # fila glioma
    assert "roc_auc_macro" not in m  # sin probabilidades no hay ROC-AUC


def test_compute_metrics_roc_auc_perfecto():
    """Si la clase real siempre tiene la probabilidad más alta, el ROC-AUC es 1."""
    y_true = [0, 1, 2, 3]
    y_proba = np.eye(4) * 0.7 + 0.075  # 0.775 en la clase real, 0.075 en el resto
    m = compute_metrics(y_true, y_proba.argmax(axis=1), y_proba)
    assert m["roc_auc_macro"] == 1.0
