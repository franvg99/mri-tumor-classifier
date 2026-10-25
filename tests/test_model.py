"""Tests unitarios para src/models/architecture.py y evaluate.py"""

import pytest
import torch

from src.models.architecture import build_model


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


def test_compute_metrics_placeholder():
    """TODO: implementar en fase de modelado (Bimestre 1)."""
    pytest.skip("Pendiente de implementación")
