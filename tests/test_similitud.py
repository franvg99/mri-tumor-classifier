"""Tests unitarios para src/data/similitud.py"""

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from src.data.similitud import UMBRAL_COPIA, similitud_maxima


def imagen_de_prueba(semilla):
    """Imagen suave y aleatoria: semillas distintas dan imágenes sin relación."""
    ruido = np.random.default_rng(semilla).integers(60, 180, size=(8, 8), dtype=np.uint8)
    return Image.fromarray(ruido).resize((256, 256), Image.BICUBIC)


@pytest.fixture
def original(tmp_path):
    path = tmp_path / "original.png"
    imagen_de_prueba(0).save(path)
    return path


@pytest.mark.parametrize(
    "alterar",
    [
        lambda img: img,
        lambda img: ImageEnhance.Brightness(img).enhance(1.2),
        lambda img: ImageEnhance.Contrast(img).enhance(0.7),
        lambda img: img.resize((200, 200)),
    ],
    ids=["guardada de nuevo", "brillo", "contraste", "tamaño"],
)
def test_detecta_copias_alteradas(original, tmp_path, alterar):
    """Copias que el hash del archivo no detecta."""
    copia = tmp_path / "copia.jpg"
    alterar(imagen_de_prueba(0)).save(copia, quality=85)
    assert similitud_maxima([copia], [original])[0] >= UMBRAL_COPIA


def test_no_confunde_imagenes_distintas(original, tmp_path):
    otra = tmp_path / "otra.png"
    imagen_de_prueba(1).save(otra)
    assert similitud_maxima([otra], [original])[0] < 0.9


def test_devuelve_la_mayor_similitud_de_la_referencia(original, tmp_path):
    otra = tmp_path / "otra.png"
    imagen_de_prueba(1).save(otra)
    similitudes = similitud_maxima([original, otra], [otra])
    assert similitudes.shape == (2,)
    assert similitudes[1] == pytest.approx(1, abs=1e-5)
    assert similitudes[0] < similitudes[1]
