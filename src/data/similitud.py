"""
Detección de imágenes casi idénticas por contenido.

`preprocessing.deduplicar()` compara los archivos byte a byte, por lo que no
detecta una misma imagen guardada de nuevo o con otro brillo, contraste o nitidez.
Cuando la original queda en train y la copia en test, acertar la copia no demuestra
que el modelo generalice (fuga de datos).

Método: cada imagen se reduce a 32x32 en escala de grises y se calcula la
correlación entre pares. La correlación no cambia si se altera el brillo o el
contraste global, y la reducción atenúa las diferencias de nitidez y compresión.

Limitación: no detecta cortes distintos de un mismo paciente. El dataset no
identifica a los pacientes.
"""

import numpy as np
from PIL import Image

LADO = 32
UMBRAL_COPIA = 0.999


def vectores_de_comparacion(paths) -> np.ndarray:
    """Convierte cada imagen en un vector centrado y de norma 1.

    Con esa forma, el producto entre dos vectores es su correlación.
    """
    vectores = []
    for path in paths:
        with Image.open(path) as imagen:
            gris = imagen.convert("L").resize((LADO, LADO), Image.BILINEAR)
        vector = np.asarray(gris, dtype=np.float32).ravel()
        vector = vector - vector.mean()
        vectores.append(vector / (np.linalg.norm(vector) + 1e-9))
    return np.stack(vectores)


def similitud_maxima(paths, paths_referencia) -> np.ndarray:
    """Para cada imagen, su correlación con la más parecida de la referencia."""
    return (vectores_de_comparacion(paths) @ vectores_de_comparacion(paths_referencia).T).max(axis=1)
