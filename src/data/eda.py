"""
Helpers de análisis exploratorio (EDA) del Brain Tumor MRI Dataset.

Se usan desde `notebooks/01_eda.ipynb`, que corre sobre el runtime de Colab
durante el Bimestre 1.

Todo lo que vive acá es deliberadamente agnóstico del framework de deep
learning: sólo PIL, numpy, pandas y matplotlib. El EDA es justamente lo único
sustancial que se puede hacer antes de resolver el TBD de PyTorch vs
TensorFlow, así que este módulo no debe importar ninguno de los dos.

Salida esperada del EDA (insumo para decisiones que hoy están en TBD):
- distribución por clase -> si hace falta balancear o pesar la loss
- resoluciones -> valor definitivo de `data.image_size` en config/config.yaml
- estadísticas de intensidad y duplicados -> qué tan agresivo puede ser el
  augmentation y si hay fuga de datos entre splits
"""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CLASES = ("glioma", "meningioma", "pituitary", "notumor")

# Nombres legibles para gráficos y reportes (los lee el tutor, no sólo nosotros)
ETIQUETAS = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "pituitary": "Pituitario",
    "notumor": "Sin tumor",
}

EXTENSIONES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def listar_imagenes(root_dir, clases=CLASES) -> pd.DataFrame:
    """Recorre el dataset y devuelve un DataFrame con una fila por imagen.

    Soporta las dos formas en que puede venir el dataset de Kaggle:
    `<root>/<clase>/imagen.jpg` y `<root>/<split>/<clase>/imagen.jpg`
    (el de Kaggle viene con `Training/` y `Testing/`).

    Columnas: `path`, `archivo`, `clase`, `split_origen`, `peso_kb`.
    `split_origen` es la carpeta Training/Testing del dataset original; NO es
    el split que vamos a usar nosotros, que se genera después de forma
    estratificada en `preprocessing.split_dataset()`.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio: {root}")

    clases_set = {c.lower() for c in clases}
    filas = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONES:
            continue

        # La clase es la carpeta ancestro más cercana cuyo nombre matchea una clase conocida.
        partes = [p.lower() for p in path.relative_to(root).parts[:-1]]
        clase = next((p for p in reversed(partes) if p in clases_set), None)
        if clase is None:
            continue

        idx = len(partes) - 1 - partes[::-1].index(clase)
        split_origen = partes[idx - 1] if idx > 0 else None

        filas.append(
            {
                "path": str(path),
                "archivo": path.name,
                "clase": clase,
                "split_origen": split_origen,
                "peso_kb": round(path.stat().st_size / 1024, 1),
            }
        )

    if not filas:
        raise ValueError(
            f"No se encontró ninguna imagen de las clases {sorted(clases_set)} bajo {root}"
        )

    return pd.DataFrame(filas)


def resumen_por_clase(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo y porcentaje de imágenes por clase, más el peso medio en disco.

    Responde la primera pregunta del EDA: ¿el dataset está balanceado? Un
    desbalance fuerte cambia la estrategia de entrenamiento (class weights o
    sampling), y acá importa el doble porque la métrica prioritaria es recall.
    """
    resumen = (
        df.groupby("clase")
        .agg(imagenes=("path", "count"), peso_kb_medio=("peso_kb", "mean"))
        .reindex([c for c in CLASES if c in set(df["clase"])])
    )
    resumen["porcentaje"] = (resumen["imagenes"] / len(df) * 100).round(1)
    resumen["peso_kb_medio"] = resumen["peso_kb_medio"].round(1)
    resumen.index = [ETIQUETAS.get(c, c) for c in resumen.index]
    return resumen[["imagenes", "porcentaje", "peso_kb_medio"]]


def inspeccionar_metadatos(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega ancho, alto, modo de color y relación de aspecto de cada imagen.

    Es rápido sobre las 7.023 imágenes porque PIL lee el header sin decodificar
    los píxeles: `Image.open()` es lazy y `.size` / `.mode` no cargan el bitmap.
    """
    anchos, altos, modos = [], [], []

    for path in df["path"]:
        try:
            with Image.open(path) as img:
                anchos.append(img.width)
                altos.append(img.height)
                modos.append(img.mode)
        except Exception:
            # Una imagen corrupta es un hallazgo del EDA, no un motivo para cortar.
            anchos.append(np.nan)
            altos.append(np.nan)
            modos.append("ERROR")

    out = df.copy()
    out["ancho"] = anchos
    out["alto"] = altos
    out["modo"] = modos
    out["lado_menor"] = out[["ancho", "alto"]].min(axis=1)
    out["aspect_ratio"] = (out["ancho"] / out["alto"]).round(3)
    out["cuadrada"] = out["ancho"] == out["alto"]
    return out


def estadisticas_intensidad(df: pd.DataFrame, muestra=400, seed=42) -> pd.DataFrame:
    """Estadísticas de píxel por imagen, sobre una submuestra estratificada por clase.

    Esto sí decodifica los píxeles, así que corre sobre una muestra y no sobre
    el dataset entero. Devuelve media, desvío, percentiles y la fracción de
    píxeles casi negros (el fondo de la MRI), que es la señal más directa de
    cuánto "aire" hay alrededor del cerebro.
    """
    if muestra is not None and muestra < len(df):
        n_por_clase = max(1, muestra // df["clase"].nunique())
        partes = [
            g.sample(min(len(g), n_por_clase), random_state=seed)
            for _, g in df.groupby("clase")
        ]
        df = pd.concat(partes).reset_index(drop=True)

    filas = []
    for _, fila in df.iterrows():
        try:
            with Image.open(fila["path"]) as img:
                # Convertimos a escala de grises: una MRI no tiene color real,
                # y así las stats son comparables entre imágenes L y RGB.
                arr = np.asarray(img.convert("L"), dtype=np.float32)
        except Exception:
            continue

        filas.append(
            {
                "path": fila["path"],
                "clase": fila["clase"],
                "media": arr.mean(),
                "desvio": arr.std(),
                "p1": np.percentile(arr, 1),
                "p99": np.percentile(arr, 99),
                "frac_fondo": float((arr < 10).mean()),  # píxeles casi negros
                "frac_saturada": float((arr > 250).mean()),
            }
        )

    return pd.DataFrame(filas).round(3)


def detectar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Busca archivos byte a byte idénticos usando MD5.

    Importa más que en un dataset cualquiera: este es una combinación de tres
    fuentes (figshare + SARTAJ + Br35H), así que puede traer la misma imagen
    repetida. Si un duplicado cae en train y su copia en test, el modelo
    aprueba el examen habiendo visto las respuestas.

    Devuelve sólo las filas que participan de algún grupo duplicado, con la
    columna `hash` para agruparlas.
    """
    hashes = []
    for path in df["path"]:
        try:
            with open(path, "rb") as f:
                hashes.append(hashlib.md5(f.read()).hexdigest())
        except Exception:
            hashes.append(None)

    out = df.copy()
    out["hash"] = hashes
    duplicados = out[out["hash"].notna() & out.duplicated("hash", keep=False)]
    return duplicados.sort_values("hash")


def recomendar_image_size(df_meta: pd.DataFrame, candidatos=(150, 224, 256)) -> pd.DataFrame:
    """Contrasta los tamaños candidatos contra la distribución real de resoluciones.

    Insumo para resolver el TBD de `data.image_size` en config/config.yaml, que
    hoy está en 150 sin justificación. Para cada candidato informa qué
    porcentaje de las imágenes habría que agrandar (upscaling, que inventa
    detalle que no existe) y cuánto se reescala en promedio.
    """
    lado = df_meta["lado_menor"].dropna()
    filas = []
    for size in candidatos:
        filas.append(
            {
                "candidato": size,
                "pct_upscaling": round((lado < size).mean() * 100, 1),
                "pct_sin_cambio": round((lado == size).mean() * 100, 1),
                "pct_downscaling": round((lado > size).mean() * 100, 1),
                "factor_medio": round((size / lado).mean(), 2),
            }
        )
    return pd.DataFrame(filas).set_index("candidato")


# --------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------

def graficar_distribucion_clases(df: pd.DataFrame, figsize=(11, 4)):
    """Barras de imágenes por clase, y el desglose Training/Testing si existe."""
    tiene_split = df["split_origen"].notna().any()
    fig, axes = plt.subplots(1, 2 if tiene_split else 1, figsize=figsize)
    axes = np.atleast_1d(axes)

    conteo = df["clase"].value_counts().reindex([c for c in CLASES if c in set(df["clase"])])
    axes[0].bar([ETIQUETAS.get(c, c) for c in conteo.index], conteo.values, color="#4C72B0")
    axes[0].set_title("Imágenes por clase")
    axes[0].set_ylabel("Cantidad")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(conteo.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    if tiene_split:
        tabla = pd.crosstab(df["clase"], df["split_origen"])
        tabla.index = [ETIQUETAS.get(c, c) for c in tabla.index]
        tabla.plot(kind="bar", stacked=True, ax=axes[1], colormap="Blues")
        axes[1].set_title("Desglose por carpeta original del dataset")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    return fig


def graficar_resoluciones(df_meta: pd.DataFrame, figsize=(13, 4)):
    """Dispersión ancho/alto, distribución del lado menor y relación de aspecto."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].scatter(df_meta["ancho"], df_meta["alto"], s=4, alpha=0.2, color="#4C72B0")
    axes[0].set_title("Ancho vs alto")
    axes[0].set_xlabel("ancho (px)")
    axes[0].set_ylabel("alto (px)")

    axes[1].hist(df_meta["lado_menor"].dropna(), bins=40, color="#4C72B0")
    axes[1].set_title("Lado menor")
    axes[1].set_xlabel("px")

    axes[2].hist(df_meta["aspect_ratio"].dropna(), bins=40, color="#4C72B0")
    axes[2].axvline(1.0, color="crimson", linestyle="--", label="cuadrada")
    axes[2].set_title("Relación de aspecto")
    axes[2].legend()

    fig.tight_layout()
    return fig


def graficar_muestras(df: pd.DataFrame, n_por_clase=4, seed=42, figsize=(11, 11)):
    """Grilla de imágenes de ejemplo: una fila por clase.

    La inspección visual no es decorativa: es lo que deja ver la orientación de
    los cortes, el contraste y cuánto fondo negro hay, que es lo que después
    limita qué augmentation es razonable aplicar.
    """
    clases = [c for c in CLASES if c in set(df["clase"])]
    fig, axes = plt.subplots(len(clases), n_por_clase, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i, clase in enumerate(clases):
        disponibles = df[df["clase"] == clase]
        muestra = disponibles.sample(min(n_por_clase, len(disponibles)), random_state=seed)
        for j, (_, fila) in enumerate(muestra.iterrows()):
            with Image.open(fila["path"]) as img:
                axes[i, j].imshow(img.convert("L"), cmap="gray")
            axes[i, j].axis("off")
            if j == 0:
                axes[i, j].set_title(ETIQUETAS.get(clase, clase), loc="left", fontsize=11)

    fig.tight_layout()
    return fig


def graficar_intensidades(df_stats: pd.DataFrame, figsize=(13, 4)):
    """Distribución de intensidad media, contraste y fondo negro, por clase."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    clases = [c for c in CLASES if c in set(df_stats["clase"])]

    for col, ax, titulo in zip(
        ["media", "desvio", "frac_fondo"],
        axes,
        ["Intensidad media", "Contraste (desvío)", "Fracción de fondo negro"],
    ):
        ax.boxplot([df_stats[df_stats["clase"] == c][col].values for c in clases])
        # set_xticklabels en vez del parámetro `labels`/`tick_labels` de boxplot:
        # ese kwarg cambió de nombre entre versiones de matplotlib y rompería
        # según la versión que tenga el runtime de Colab.
        ax.set_xticks(range(1, len(clases) + 1))
        ax.set_xticklabels([ETIQUETAS.get(c, c) for c in clases])
        ax.set_title(titulo)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    return fig
