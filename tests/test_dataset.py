"""Tests unitarios para src/data/dataset.py"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import CLASES, MRIDataset

TRANSFORM = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


@pytest.fixture
def df_imagenes(tmp_path):
    """Una imagen sintética en escala de grises por clase, como las MRI reales."""
    filas = []
    for clase in CLASES:
        path = tmp_path / f"{clase}.png"
        Image.fromarray(np.zeros((32, 32), dtype=np.uint8), mode="L").save(path)
        filas.append({"path": str(path), "clase": clase})
    return pd.DataFrame(filas)


def test_muestra_es_tensor_rgb_y_etiqueta_numerica(df_imagenes):
    dataset = MRIDataset(df_imagenes, TRANSFORM)

    assert len(dataset) == 4
    for i, clase in enumerate(CLASES):
        imagen, etiqueta = dataset[i]
        assert imagen.shape == (3, 224, 224)  # gris convertido a 3 canales
        assert etiqueta == CLASES.index(clase)


def test_dataloader_arma_batches(df_imagenes):
    imagenes, etiquetas = next(iter(DataLoader(MRIDataset(df_imagenes, TRANSFORM), batch_size=4)))

    assert imagenes.shape == (4, 3, 224, 224)
    assert etiquetas.tolist() == [0, 1, 2, 3]


def test_rechaza_clases_desconocidas(df_imagenes):
    df_imagenes.loc[0, "clase"] = "Glioma"  # mayúscula: no matchea

    with pytest.raises(ValueError, match="Clases desconocidas"):
        MRIDataset(df_imagenes, TRANSFORM)
