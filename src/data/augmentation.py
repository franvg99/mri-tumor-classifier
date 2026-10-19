"""
Pipelines de transformación de imágenes MRI (torchvision).

Cada pipeline recibe una imagen PIL y devuelve un tensor `(3, 224, 224)`
normalizado, listo para EfficientNet-B0. Se le pasan a `MRIDataset` como
`transform`.

- `get_train_augmentation_pipeline()`: con augmentation, SOLO para train.
- `get_eval_pipeline()`: sin augmentation, para val, test e inferencia. Es la
  única definición del preprocesamiento de evaluación: `inference.py` debe
  importarla en lugar de reimplementar el resize y la normalización, para que
  el endpoint reciba exactamente lo mismo que vio el modelo al entrenar.

Augmentation aplicado en train:
- Rotaciones leves (±10 grados) y zoom leve (±10%)
- Ajustes de brillo/contraste (±40%, obligatorio, ver justificación abajo)
- SIN flip horizontal (excluido a propósito, ver justificación abajo)

Se evitan transformaciones agresivas (ej. deformaciones elásticas fuertes)
que podrían distorsionar features clínicamente relevantes.

Justificación de las decisiones no obvias (EDA, notebook 01, secciones 4 y 5,
y docs/documentacion_tecnica.md):

- **Sin flip horizontal:** el dataset mezcla cortes axial/sagital/coronal sin
  etiquetar. Un flip horizontal es válido en axial/coronal pero anatómicamente
  inválido en sagital (invierte adelante-atrás). Sin forma barata de detectar
  la orientación por imagen, se descarta el flip antes que arriesgar corromper
  el subconjunto sagital.
- **Brillo/contraste obligatorio, y más fuerte que la geometría:** hay
  diferencias sistemáticas de intensidad entre clases que coinciden con que el
  dataset combina 3 fuentes distintas (glioma=figshare, meningioma/pituitario≈
  SARTAJ, notumor=100% Br35H). El factor 0.4 hace que los rangos de brillo de
  las 4 clases se superpongan (notumor es ~1.8 veces más brillante que glioma
  en promedio). Escalar la intensidad global no altera la anatomía, por eso
  puede ser más fuerte que la geometría. Limitación medida: reduce el atajo
  solo levemente (un clasificador de brillo/desvío pasa de 63% a 61%); se
  controla aparte con `evaluate.brightness_stress_test()`.
"""

from torchvision import transforms

IMAGE_SIZE = 224

# Media y desvío por canal de ImageNet: EfficientNet-B0 fue preentrenada con
# imágenes normalizadas con estos valores, y espera recibir la misma escala.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_augmentation_pipeline():
    """Pipeline de train: resize + augmentation + tensor normalizado.

    El resize va primero para que el augmentation opere sobre imágenes de 224 px
    en vez de 512 px (se entrena en CPU, sin GPU).
    """
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # Las esquinas que deja la rotación se rellenan con negro (fill=0),
            # igual que el fondo de la MRI.
            transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), fill=0),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_pipeline():
    """Pipeline sin augmentation, para val, test e inferencia (solo resize + normalize)."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
