"""
Dataset de PyTorch para las imágenes MRI.

Sirve las imágenes al modelo de a una, a demanda: recibe el DataFrame de
`eda.listar_imagenes()` (ya deduplicado y filtrado por split) y, para cada
índice, abre la imagen, le aplica la transformación recibida y la devuelve
junto con su etiqueta numérica.

La misma clase se usa para train, val y test; lo único que cambia es la
transformación que se le pasa (con augmentation en train, sin augmentation en
val/test — ver `augmentation.py`).
"""

from PIL import Image
from torch.utils.data import Dataset

# El orden define el índice de cada clase en la salida del modelo.
# Tiene que coincidir con `data.classes` de config/config.yaml.
CLASES = ("glioma", "meningioma", "pituitary", "notumor")
CLASE_A_INDICE = {clase: i for i, clase in enumerate(CLASES)}


class MRIDataset(Dataset):
    """Dataset de imágenes MRI con su clase de tumor.

    Args:
        df: DataFrame con al menos las columnas `path` y `clase`, en el formato
            que devuelve `eda.listar_imagenes()`.
        transform: transformación que convierte la imagen PIL en un tensor
            `(3, 224, 224)` (pipeline de `augmentation.py`).
    """

    def __init__(self, df, transform):
        desconocidas = set(df["clase"]) - set(CLASES)
        if desconocidas:
            raise ValueError(f"Clases desconocidas en el DataFrame: {sorted(desconocidas)}")

        self.paths = df["path"].tolist()
        self.etiquetas = df["clase"].map(CLASE_A_INDICE).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # Las MRI vienen en escala de grises (o RGB con los 3 canales iguales),
        # pero EfficientNet-B0 fue preentrenada sobre ImageNet con 3 canales.
        # convert("RGB") replica el canal gris en los 3.
        imagen = Image.open(self.paths[i]).convert("RGB")
        return self.transform(imagen), self.etiquetas[i]
