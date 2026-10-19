"""Tests unitarios para src/data/preprocessing.py"""

import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.preprocessing import deduplicar, split_dataset

CLASES = ("glioma", "meningioma", "pituitary", "notumor")
POR_CLASE = 50


@pytest.fixture
def df_imagenes(tmp_path):
    """50 imágenes sintéticas de 8x8 por clase, todas distintas entre sí.

    Mismo formato de DataFrame que `eda.listar_imagenes()` (columnas `path` y `clase`).
    """
    rng = np.random.default_rng(0)
    filas = []
    for clase in CLASES:
        for i in range(POR_CLASE):
            path = tmp_path / f"{clase}_{i}.png"
            Image.fromarray(rng.integers(0, 256, (8, 8), dtype=np.uint8)).save(path)
            filas.append({"path": str(path), "clase": clase})
    return pd.DataFrame(filas)


def _agregar_copia(df, fila, clase=None):
    """Copia el archivo de `fila` con otro nombre y lo agrega al DataFrame."""
    origen = df.loc[fila, "path"]
    destino = origen.replace(".png", "_copia.png")
    shutil.copy(origen, destino)
    nueva = {"path": destino, "clase": clase or df.loc[fila, "clase"]}
    return pd.concat([df, pd.DataFrame([nueva])], ignore_index=True)


def test_resize_image_placeholder():
    """TODO: implementar en fase de modelado (Bimestre 1)."""
    pytest.skip("Pendiente de implementación")


def test_normalize_image_placeholder():
    """TODO: implementar en fase de modelado (Bimestre 1)."""
    pytest.skip("Pendiente de implementación")


def test_deduplicar_conserva_una_copia(df_imagenes):
    df = _agregar_copia(_agregar_copia(df_imagenes, 0), 60)

    resultado = deduplicar(df)

    assert len(resultado) == len(df_imagenes)
    assert resultado["path"].is_unique


def test_deduplicar_rechaza_duplicado_con_clases_distintas(df_imagenes):
    df = _agregar_copia(df_imagenes, 0, clase="notumor")  # la fila 0 es glioma

    with pytest.raises(ValueError, match="clases distintas"):
        deduplicar(df)


def test_split_proporciones_estratificadas(df_imagenes):
    df_train, df_val, df_test = split_dataset(df_imagenes)

    # 50 por clase con 70/10/20 -> 35/5/10 exactos en cada clase
    for df_split, esperado in [(df_train, 35), (df_val, 5), (df_test, 10)]:
        conteo = df_split["clase"].value_counts()
        assert set(conteo.index) == set(CLASES)
        assert (conteo == esperado).all()


def test_split_sin_solapamiento(df_imagenes):
    df_train, df_val, df_test = split_dataset(df_imagenes)

    train, val, test = (set(d["path"]) for d in (df_train, df_val, df_test))
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train | val | test) == len(df_imagenes)


def test_split_deduplica_antes_de_dividir(df_imagenes):
    df = _agregar_copia(df_imagenes, 0)

    df_train, df_val, df_test = split_dataset(df)

    assert len(df_train) + len(df_val) + len(df_test) == len(df_imagenes)


def test_split_reproducible_con_otro_orden_de_entrada(df_imagenes):
    split_a = split_dataset(df_imagenes)
    split_b = split_dataset(df_imagenes.sample(frac=1, random_state=1))

    for a, b in zip(split_a, split_b):
        assert a["path"].tolist() == b["path"].tolist()


def test_split_rechaza_proporciones_que_no_suman_uno(df_imagenes):
    with pytest.raises(ValueError, match="sumar 1"):
        split_dataset(df_imagenes, train_split=0.8, val_split=0.1, test_split=0.2)
