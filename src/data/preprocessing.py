"""
Preprocesamiento de imágenes MRI: resize y normalización.

Prototipado local (Bimestre 1); este módulo se porta luego al Studio
Notebook de SageMaker sin modificaciones (mismo código, distinto
entorno de ejecución).
"""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split


def resize_image(image, target_size: int = 224):
    """Redimensiona una imagen al tamaño de entrada del modelo.

    TODO: implementar en fase de modelado (Bimestre 1).
    """
    raise NotImplementedError


def normalize_image(image):
    """Normaliza los valores de píxel (0-255 -> rango esperado por el backbone preentrenado).

    TODO: implementar en fase de modelado (Bimestre 1).
    """
    raise NotImplementedError


def _hash_md5(path) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def deduplicar(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina las imágenes byte a byte idénticas, conservando una copia por grupo.

    Misma lógica de detección que `eda.detectar_duplicados()` (hash MD5 del
    archivo). El DataFrame se ordena por `path` antes de descartar, de modo que la
    copia conservada no depende del orden de entrada y el split posterior es
    reproducible.

    Si un mismo archivo aparece con dos clases distintas, levanta `ValueError`: no
    hay forma de saber cuál etiqueta es la correcta. El EDA no encontró ningún caso.
    """
    out = df.sort_values("path").copy()
    out["hash"] = out["path"].map(_hash_md5)

    clases_por_hash = out.groupby("hash")["clase"].nunique()
    if (clases_por_hash > 1).any():
        raise ValueError(
            f"{(clases_por_hash > 1).sum()} imágenes duplicadas aparecen con clases distintas"
        )

    return out.drop_duplicates("hash", keep="first").drop(columns="hash").reset_index(drop=True)


def split_dataset(df: pd.DataFrame, train_split=0.7, val_split=0.1, test_split=0.2, seed=42):
    """Deduplica y divide el dataset en train/val/test, estratificado por clase.

    Recibe el DataFrame de `eda.listar_imagenes()` (columnas `path` y `clase`) y
    devuelve `(df_train, df_val, df_test)`, listos para pasarle a `MRIDataset`.

    Ignora el split Training/Testing original de Kaggle (`split_origen` en el DataFrame
    del EDA) — es un split ajeno, no el del proyecto.

    Deduplica ANTES de dividir (hallazgo del EDA, sección 6: 340 duplicados exactos,
    4.72%): si no, una imagen podría quedar con una copia en train y otra en test,
    inflando las métricas de test.

    Proporciones 70/10/20: el test es más grande que el val porque el resultado que
    se reporta es el recall por clase, y con ~350 imágenes por clase en test (en vez
    de ~175) el margen de error de cada recall baja de ±4.4 a ±3.1 puntos. El costo
    en train es bajo porque, con el backbone congelado, solo se entrena la capa final.

    `train_test_split` solo divide en dos, así que se aplica dos veces: primero
    train contra el resto, y después el resto en val y test. Las dos veces con
    `stratify` y la misma `seed`.
    """
    if abs(train_split + val_split + test_split - 1) > 1e-6:
        raise ValueError("train_split + val_split + test_split debe sumar 1")

    df = deduplicar(df)

    df_train, df_resto = train_test_split(
        df, train_size=train_split, stratify=df["clase"], random_state=seed
    )
    df_val, df_test = train_test_split(
        df_resto,
        test_size=test_split / (val_split + test_split),
        stratify=df_resto["clase"],
        random_state=seed,
    )

    return tuple(d.reset_index(drop=True) for d in (df_train, df_val, df_test))
