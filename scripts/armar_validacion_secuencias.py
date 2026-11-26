"""
Arma la muestra para la validación de secuencias por un profesional.

Objetivo: comprobar si el modelo decide "tumor o sano" según el tipo de secuencia
(T1, T1 con contraste, T2, FLAIR) en lugar de según el tumor. Si usara ese atajo,
dejaría huellas en sus errores (ver `docs/documentacion_tecnica.md`, punto 6):

- tumores predichos como sanos con aspecto T2/FLAIR ("T2/FLAIR -> sano");
- imágenes sanas predichas como tumor con aspecto T1 ("T1 -> tumor").

La muestra tiene 50 imágenes del test sin copias en train:

- los tumores predichos como sanos, una imagen por grupo de copias (10);
- todas las sanas que el modelo predijo como tumor (4);
- sanas acertadas elegidas al azar hasta completar 40 sanas.

Las sanas erradas están sobrerrepresentadas respecto del test (4 de 40 contra 4 de
115). Por eso el resultado se informa como conteo y no como porcentaje de acierto.

La evaluación es a ciegas: las imágenes se mezclan y se identifican con un número
(img_01 a img_50). La tabla que vincula cada número con su archivo, su clase y la
predicción del modelo (`clave.csv`) queda en `data/validacion/`, que no se versiona.

El script también genera `validacion_secuencias.html`, una página autocontenida (las
imágenes van incrustadas) para enviar al profesional. Contiene solo el número de cada
imagen; el diseño está en `scripts/plantilla_validacion_secuencias.html`.

Uso:
    python scripts/armar_validacion_secuencias.py --data-dir <carpeta del dataset>
"""

import argparse
import base64
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.augmentation import get_eval_pipeline  # noqa: E402
from src.data.dataset import CLASES, MRIDataset  # noqa: E402
from src.data.eda import listar_imagenes  # noqa: E402
from src.data.preprocessing import split_dataset  # noqa: E402
from src.data.similitud import UMBRAL_COPIA, similitud_maxima, vectores_de_comparacion  # noqa: E402
from src.models.evaluate import cargar_modelo, predecir  # noqa: E402

N_SANAS = 40
LADO_MAXIMO = 512  # px; las imágenes más grandes se reducen a este tamaño en el HTML
PLANTILLA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "plantilla_validacion_secuencias.html")
MARCADOR = "/*__IMAGENES__*/[]"


def predecir_test_sin_copias(data_dir, model_dir, seed=42) -> pd.DataFrame:
    """Predice el test excluyendo las imágenes con copia en train.

    Usa el mismo split y el mismo criterio de copia que `evaluate.main()`, de modo
    que los errores coinciden con los reportados en `docs/model_card.md`.
    """
    df_train, _, df_test = split_dataset(listar_imagenes(data_dir), seed=seed)
    loader = DataLoader(MRIDataset(df_test, get_eval_pipeline()), batch_size=64)
    _, y_pred, y_proba = predecir(cargar_modelo(model_dir), loader)

    nuevas = similitud_maxima(df_test["path"], df_train["path"]) < UMBRAL_COPIA
    df = df_test.assign(
        prediccion=[CLASES[i] for i in y_pred],
        seguridad=y_proba.max(axis=1),
        lado=[min(Image.open(p).size) for p in df_test["path"]],
    )
    return df[nuevas].reset_index(drop=True)


def una_por_grupo_de_copias(paths) -> np.ndarray:
    """Máscara que conserva la primera imagen de cada grupo de copias casi idénticas."""
    similitudes = vectores_de_comparacion(paths) @ vectores_de_comparacion(paths).T
    repetida = np.triu(similitudes >= UMBRAL_COPIA, k=1).any(axis=0)
    return ~repetida


def elegir_muestra(df, n_sanas=N_SANAS, seed=42) -> pd.DataFrame:
    """Elige la muestra, la mezcla y le asigna un número a cada imagen."""
    tumor = df["clase"] != "notumor"
    predicha_sana = df["prediccion"] == "notumor"

    graves = df[tumor & predicha_sana]
    graves = graves[una_por_grupo_de_copias(graves["path"])].assign(grupo="tumor_predicho_sano")
    sanas_erradas = df[~tumor & ~predicha_sana].assign(grupo="sana_errada")
    sanas_acertadas = df[~tumor & predicha_sana].sample(
        n=n_sanas - len(sanas_erradas), random_state=seed
    ).assign(grupo="sana_acertada")

    muestra = pd.concat([graves, sanas_erradas, sanas_acertadas])
    muestra = muestra.sample(frac=1, random_state=seed).reset_index(drop=True)
    muestra.insert(0, "id", [f"img_{i:02d}" for i in range(1, len(muestra) + 1)])
    return muestra[["id", "grupo", "clase", "prediccion", "seguridad", "lado", "path"]]


def imagen_como_texto(path, lado_maximo=LADO_MAXIMO) -> str:
    """Convierte una imagen en texto (data URI) para incrustarla en el HTML.

    Se vuelve a guardar como JPEG en escala de grises: así se descartan los metadatos
    del archivo original y el HTML queda liviano.
    """
    with Image.open(path) as imagen:
        gris = imagen.convert("L")
    gris.thumbnail((lado_maximo, lado_maximo))
    buffer = io.BytesIO()
    gris.save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def generar_html(muestra, ruta_salida, plantilla=PLANTILLA):
    """Genera la página de validación con las imágenes de la muestra incrustadas.

    Solo se incrustan el número y la imagen: ni la clase ni la predicción llegan al HTML.
    """
    with open(plantilla, encoding="utf-8") as f:
        html = f.read()
    if MARCADOR not in html:
        raise ValueError(f"La plantilla no contiene el marcador {MARCADOR}")

    imagenes = [{"id": fila.id, "src": imagen_como_texto(fila.path)} for fila in muestra.itertuples()]
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(html.replace(MARCADOR, json.dumps(imagenes)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Muestra para la validación de secuencias")
    parser.add_argument("--data-dir", required=True,
                        help="Carpeta del dataset de Kaggle (con las 4 clases).")
    parser.add_argument("--model-dir", default="models", help="Carpeta con model.pth.")
    parser.add_argument("--output-dir", default=os.path.join("data", "validacion"),
                        help="Dónde se guarda clave.csv (no versionado).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    muestra = elegir_muestra(predecir_test_sin_copias(args.data_dir, args.model_dir, args.seed),
                             seed=args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    ruta_clave = os.path.join(args.output_dir, "clave.csv")
    muestra.to_csv(ruta_clave, index=False)

    ruta_html = os.path.join(args.output_dir, "validacion_secuencias.html")
    generar_html(muestra, ruta_html)

    print(f"Muestra: {len(muestra)} imágenes -> {ruta_clave}")
    print(muestra["grupo"].value_counts().to_string())
    print(f"Página para el profesional: {ruta_html} ({os.path.getsize(ruta_html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
