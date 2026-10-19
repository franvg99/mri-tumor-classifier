"""Tests unitarios para src/data/augmentation.py"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.data.augmentation import get_eval_pipeline, get_train_augmentation_pipeline


@pytest.fixture
def imagen_mri():
    """Imagen sintética de 512x512 en RGB con un "cerebro" gris sobre fondo negro."""
    arr = np.zeros((512, 512), dtype=np.uint8)
    arr[128:384, 128:384] = 120
    return Image.fromarray(arr).convert("RGB")


@pytest.mark.parametrize("pipeline", [get_train_augmentation_pipeline, get_eval_pipeline])
def test_salida_es_tensor_normalizado_de_224(pipeline, imagen_mri):
    tensor = pipeline()(imagen_mri)

    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    # Normalizado con ImageNet: el fondo negro queda en valores negativos
    assert tensor.min() < 0


def test_eval_es_determinista(imagen_mri):
    pipeline = get_eval_pipeline()

    assert torch.equal(pipeline(imagen_mri), pipeline(imagen_mri))


def test_train_varia_en_cada_llamada(imagen_mri):
    torch.manual_seed(0)
    pipeline = get_train_augmentation_pipeline()

    assert not torch.equal(pipeline(imagen_mri), pipeline(imagen_mri))


def test_train_no_incluye_flip():
    nombres = [type(t).__name__ for t in get_train_augmentation_pipeline().transforms]

    assert not any("Flip" in n for n in nombres)
