"""
Evaluación del modelo.

Métricas priorizadas por el contexto sanitario del problema (un falso
negativo — no detectar un tumor real — es más costoso que un falso
positivo):

- Recall / Sensibilidad (por clase)
- Precisión y F1-Score (por clase y macro)
- Matriz de confusión
- ROC-AUC (one-vs-rest, multiclase)
- Imágenes con tumor clasificadas como "notumor" (el falso negativo
  clínicamente más grave)

El conjunto de test se usa solo para medir: ninguna decisión sobre el modelo
(época, hiperparámetros, umbrales) se toma mirando su resultado, porque dejaría
de ser una estimación limpia del desempeño.

Uso local:
    python -m src.models.evaluate --data-dir <carpeta del dataset de Kaggle>
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: el script corre sin pantalla
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.augmentation import get_eval_pipeline
from src.data.dataset import CLASE_A_INDICE, CLASES, MRIDataset
from src.data.eda import ETIQUETAS, listar_imagenes
from src.data.preprocessing import split_dataset
from src.models.architecture import build_model

INDICE_NOTUMOR = CLASE_A_INDICE["notumor"]


def compute_metrics(y_true, y_pred, y_proba=None):
    """Calcula recall, precisión, F1, matriz de confusión y ROC-AUC.

    Args:
        y_true: etiquetas reales (índices de `CLASES`).
        y_pred: etiquetas predichas (índices de `CLASES`).
        y_proba: probabilidades por clase, shape `(n, 4)`. Opcional; sin ellas
            no se calcula el ROC-AUC.

    Returns:
        Diccionario con las métricas por clase, sus promedios macro, la
        accuracy, la matriz de confusión (filas: clase real; columnas: clase
        predicha) y la cantidad de imágenes con tumor predichas como "notumor".
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    indices = list(range(len(CLASES)))

    precision, recall, f1, soporte = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    por_clase = {
        clase: {
            "recall": float(recall[i]),
            "precision": float(precision[i]),
            "f1": float(f1[i]),
            "soporte": int(soporte[i]),
        }
        for i, clase in enumerate(CLASES)
    }

    es_tumor = y_true != INDICE_NOTUMOR
    metricas = {
        "por_clase": por_clase,
        "recall_macro": float(recall.mean()),
        "precision_macro": float(precision.mean()),
        "f1_macro": float(f1.mean()),
        "accuracy": float((y_true == y_pred).mean()),
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=indices).tolist(),
        "tumores_como_sanos": int((es_tumor & (y_pred == INDICE_NOTUMOR)).sum()),
        "tumores_total": int(es_tumor.sum()),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        for i, clase in enumerate(CLASES):
            # One-vs-rest: la clase i contra todas las demás juntas.
            por_clase[clase]["roc_auc"] = float(roc_auc_score(y_true == i, y_proba[:, i]))
        metricas["roc_auc_macro"] = float(
            np.mean([por_clase[c]["roc_auc"] for c in CLASES])
        )

    return metricas


def predecir(model, loader):
    """Corre el modelo sobre un loader completo.

    Returns:
        `(y_true, y_pred, y_proba)`: etiquetas reales, clase predicha y
        probabilidades por clase (softmax de los logits).
    """
    model.eval()
    y_true, y_proba = [], []
    with torch.no_grad():
        for imagenes, etiquetas in loader:
            y_proba.append(torch.softmax(model(imagenes), dim=1))
            y_true.extend(etiquetas.tolist())

    y_proba = torch.cat(y_proba).numpy()
    return np.array(y_true), y_proba.argmax(axis=1), y_proba


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Genera la matriz de confusión como imagen, para incluir en la documentación final.

    Returns:
        La figura de matplotlib (quien la llama decide si la guarda o la muestra).
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        display_labels=class_names,
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set(title="Matriz de confusión (test)", xlabel="Predicción", ylabel="Clase real")
    fig.tight_layout()
    return fig


def _pipeline_con_brillo(factor):
    """Pipeline de evaluación precedido por un cambio de brillo de `factor`."""
    return transforms.Compose(
        [transforms.Lambda(lambda img: TF.adjust_brightness(img, factor)), get_eval_pipeline()]
    )


def brightness_stress_test(model, df_test, factores=(0.5, 0.75, 1.0, 1.25, 1.5, 1.8)):
    """Mide si el modelo usa el brillo como atajo para predecir la clase.

    Multiplica el brillo de las imágenes de test por cada factor y reporta, para
    cada uno:

    - `tumores_como_sanos`: fracción de imágenes con tumor predichas como
      "notumor". Si crece al aclarar las imágenes, el modelo se apoya en el
      brillo (señal de la fuente Br35H, no de la patología).
    - `recall_notumor`: fracción de imágenes sin tumor que se siguen prediciendo
      como "notumor". Si cae al oscurecerlas, es el mismo atajo visto desde la
      otra clase.

    Motivación: un clasificador que solo ve brillo medio y desvío acierta el 63%
    (azar: 25%), y como test proviene de las mismas fuentes que train, la métrica
    de test no puede detectar el atajo. Ver docs/documentacion_tecnica.md,
    sección 4 de los hallazgos del EDA.

    Rango de factores: el augmentation de train ya varía el brillo entre 0.6 y
    1.4, por lo que dentro de ese rango se espera robustez por construcción. El
    rango se extiende hasta 1.8, la relación de brillo medio entre "notumor" y
    glioma medida en el EDA. En los factores altos, parte de los píxeles se
    satura en blanco.

    Returns:
        DataFrame con una fila por factor.
    """
    filas = []
    for factor in factores:
        loader = DataLoader(MRIDataset(df_test, _pipeline_con_brillo(factor)), batch_size=64)
        y_true, y_pred, _ = predecir(model, loader)
        es_tumor = y_true != INDICE_NOTUMOR
        filas.append(
            {
                "factor": factor,
                "tumores_como_sanos": float((y_pred[es_tumor] == INDICE_NOTUMOR).mean()),
                "recall_notumor": float((y_pred[~es_tumor] == INDICE_NOTUMOR).mean()),
                "accuracy": float((y_true == y_pred).mean()),
            }
        )
    return pd.DataFrame(filas)


def plot_brightness_stress_test(df_estres):
    """Grafica el resultado de `brightness_stress_test()` en función del factor."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df_estres["factor"], df_estres["tumores_como_sanos"], marker="o",
            label="Tumores predichos como sin tumor")
    ax.plot(df_estres["factor"], df_estres["recall_notumor"], marker="o",
            label="Sin tumor reconocidos como sin tumor")
    ax.axvspan(0.6, 1.4, color="grey", alpha=0.15, label="Rango del augmentation de train")
    ax.set(title="Prueba de estrés de brillo (test)", xlabel="Factor de brillo",
           ylabel="Fracción", ylim=(-0.02, 1.02))
    ax.legend()
    fig.tight_layout()
    return fig


def cargar_modelo(model_dir):
    """Reconstruye la arquitectura y le carga los pesos entrenados de `model.pth`."""
    model = build_model(num_classes=len(CLASES), pretrained=False)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth")))
    return model


def main(argv=None):
    """Evalúa el modelo entrenado sobre el conjunto de test y guarda los resultados."""
    parser = argparse.ArgumentParser(description="Evaluación del clasificador MRI sobre test")
    parser.add_argument("--data-dir", required=True,
                        help="Carpeta del dataset de Kaggle (con las 4 clases).")
    parser.add_argument("--model-dir", default="models", help="Carpeta con model.pth.")
    parser.add_argument("--output-dir", default="models",
                        help="Dónde se guardan las métricas y la matriz de confusión.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    # Mismo split que en train.py (misma seed): test nunca se usó para entrenar ni elegir.
    _, _, df_test = split_dataset(listar_imagenes(args.data_dir), seed=args.seed)
    loader_test = DataLoader(MRIDataset(df_test, get_eval_pipeline()), batch_size=64)

    model = cargar_modelo(args.model_dir)
    y_true, y_pred, y_proba = predecir(model, loader_test)
    metricas = compute_metrics(y_true, y_pred, y_proba)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metricas_test.json"), "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    fig = plot_confusion_matrix(y_true, y_pred, [ETIQUETAS[c] for c in CLASES])
    fig.savefig(os.path.join(args.output_dir, "matriz_confusion_test.png"), dpi=120)
    plt.close(fig)

    print(f"test: {len(y_true)} imágenes")
    for clase, m in metricas["por_clase"].items():
        print(
            f"{clase:<11} recall {m['recall']:.3f} | precisión {m['precision']:.3f} | "
            f"F1 {m['f1']:.3f} | ROC-AUC {m['roc_auc']:.3f} | n={m['soporte']}"
        )
    print(
        f"macro: recall {metricas['recall_macro']:.3f} | precisión {metricas['precision_macro']:.3f} | "
        f"F1 {metricas['f1_macro']:.3f} | ROC-AUC {metricas['roc_auc_macro']:.3f}"
    )
    print(f"tumores como sanos: {metricas['tumores_como_sanos']}/{metricas['tumores_total']}")

    df_estres = brightness_stress_test(model, df_test)
    df_estres.to_csv(os.path.join(args.output_dir, "estres_brillo_test.csv"), index=False)
    fig = plot_brightness_stress_test(df_estres)
    fig.savefig(os.path.join(args.output_dir, "estres_brillo_test.png"), dpi=120)
    plt.close(fig)

    print("\nPrueba de estrés de brillo:")
    print(df_estres.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
