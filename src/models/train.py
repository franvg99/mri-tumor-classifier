"""
Script de entrenamiento del clasificador.

Diseñado para correr igual en dos entornos:
- Entorno local (Bimestre 1, prototipado)
- SageMaker Training Job (Bimestre 2, ml.m5.xlarge, CPU, on-demand, sin Spot)

Al portarlo a SageMaker, este archivo se pasa como entry_point del
Estimator; los hiperparámetros y paths de datos se leen desde
variables de entorno / argumentos (convención de SageMaker).

Uso local:
    python -m src.models.train --data-dir <carpeta del dataset de Kaggle>

Criterio de parada (early stopping): se entrena hasta `--epochs` épocas como
techo, y se corta antes si el recall promedio de validación no mejora durante
`--patience` épocas seguidas. Se conserva el modelo de la mejor época, no el de
la última. Se usa el recall y no la loss porque es la métrica prioritaria del
proyecto (contexto clínico: un falso negativo es el error más costoso).
"""

import argparse
import copy
import json
import os
import time

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: el script corre sin pantalla (SageMaker)
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import recall_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.augmentation import get_eval_pipeline, get_train_augmentation_pipeline
from src.data.dataset import CLASE_A_INDICE, CLASES, MRIDataset
from src.data.eda import listar_imagenes
from src.data.preprocessing import split_dataset
from src.models.architecture import build_model

INDICE_NOTUMOR = CLASE_A_INDICE["notumor"]


def parse_args(argv=None):
    """Lee hiperparámetros y rutas desde la línea de comandos.

    SageMaker pasa los hiperparámetros del Estimator como argumentos
    (`--epochs 30`) y las rutas como variables de entorno (`SM_CHANNEL_TRAIN`,
    `SM_MODEL_DIR`, `SM_OUTPUT_DATA_DIR`). En local esas variables no existen,
    por lo que las rutas se pasan como argumentos o toman el valor por defecto.
    """
    parser = argparse.ArgumentParser(description="Entrenamiento del clasificador MRI")

    # Hiperparámetros
    parser.add_argument("--epochs", type=int, default=30,
                        help="Techo de épocas; el early stopping puede cortar antes.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Épocas seguidas sin mejora del recall de val antes de cortar.")
    # Valor inicial provisorio: se confirma o ajusta con las curvas de la primera corrida.
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Procesos que cargan imágenes en paralelo (0 = el proceso principal).")

    # Rutas
    parser.add_argument("--data-dir", default=os.environ.get("SM_CHANNEL_TRAIN"),
                        help="Carpeta del dataset de Kaggle (con las 4 clases).")
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "models"),
                        help="Dónde se guarda el modelo entrenado.")
    parser.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "models"),
                        help="Dónde se guardan las curvas y el historial de métricas.")

    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("falta --data-dir (o la variable de entorno SM_CHANNEL_TRAIN)")
    return args


def preparar_dataloaders(data_dir, batch_size, seed, num_workers=0):
    """Arma los DataLoaders de train y val a partir de la carpeta del dataset.

    El split se recalcula en cada corrida con `split_dataset()`; como la `seed`
    es fija, sale idéntico en local y en SageMaker sin guardar archivos
    intermedios. El conjunto de test se descarta acá a propósito: solo lo usa
    `evaluate.py`, para que ninguna decisión del entrenamiento dependa de él.

    Returns:
        `(loader_train, loader_val)`. El de train mezcla el orden de las imágenes
        en cada época y aplica augmentation; el de val no hace ninguna de las dos
        cosas, para que todas las épocas se evalúen exactamente sobre lo mismo.
    """
    df = listar_imagenes(data_dir)
    df_train, df_val, _ = split_dataset(df, seed=seed)

    loader_train = DataLoader(
        MRIDataset(df_train, get_train_augmentation_pipeline()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    loader_val = DataLoader(
        MRIDataset(df_val, get_eval_pipeline()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader_train, loader_val


def modo_entrenamiento(model):
    """Pone el head en modo entrenamiento y deja el backbone en modo evaluación.

    `model.train()` activa el Dropout del head, que es lo buscado, pero también
    haría que las capas BatchNorm del backbone actualicen sus estadísticas
    internas con cada batch, aunque sus pesos estén congelados. Volver a poner
    `model.features` en modo evaluación mantiene el backbone realmente fijo.
    """
    model.train()
    model.features.eval()


def entrenar_una_epoca(model, loader, criterion, optimizer):
    """Recorre todo el set de train una vez, corrigiendo el head en cada batch.

    Returns:
        La loss promedio por imagen de la época.
    """
    modo_entrenamiento(model)
    loss_total = 0.0

    for imagenes, etiquetas in loader:
        optimizer.zero_grad()  # descarta los gradientes del batch anterior
        logits = model(imagenes)
        loss = criterion(logits, etiquetas)
        loss.backward()  # calcula cuánto corregir cada parámetro entrenable
        optimizer.step()  # aplica la corrección
        loss_total += loss.item() * len(etiquetas)

    return loss_total / len(loader.dataset)


def evaluar(model, loader, criterion):
    """Evalúa el modelo sobre un set completo, sin modificarlo.

    Returns:
        Diccionario con la loss promedio, el recall por clase, el recall
        promedio (macro) y la cantidad de imágenes con tumor que el modelo
        clasifica como "notumor" (el falso negativo clínicamente más grave).
    """
    model.eval()
    loss_total = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():  # sin gradientes: acá no se corrige nada
        for imagenes, etiquetas in loader:
            logits = model(imagenes)
            loss_total += criterion(logits, etiquetas).item() * len(etiquetas)
            y_true.extend(etiquetas.tolist())
            y_pred.extend(logits.argmax(dim=1).tolist())

    recall_por_clase = recall_score(
        y_true, y_pred, labels=list(range(len(CLASES))), average=None, zero_division=0
    )
    tumores_como_sanos = sum(
        1 for t, p in zip(y_true, y_pred) if t != INDICE_NOTUMOR and p == INDICE_NOTUMOR
    )

    return {
        "loss": loss_total / len(loader.dataset),
        "recall_macro": float(recall_por_clase.mean()),
        "recall_por_clase": dict(zip(CLASES, recall_por_clase.tolist())),
        "tumores_como_sanos": tumores_como_sanos,
        "tumores_total": sum(1 for t in y_true if t != INDICE_NOTUMOR),
    }


def guardar_curvas(historial, output_dir):
    """Guarda el historial por época (JSON) y el gráfico de las curvas (PNG)."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "historial.json"), "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=2, ensure_ascii=False)

    epocas = [h["epoca"] for h in historial]
    fig, (ax_loss, ax_recall) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epocas, [h["loss_train"] for h in historial], marker="o", label="train")
    ax_loss.plot(epocas, [h["val"]["loss"] for h in historial], marker="o", label="val")
    ax_loss.set(title="Loss por época", xlabel="Época", ylabel="Loss")
    ax_loss.legend()

    ax_recall.plot(epocas, [h["val"]["recall_macro"] for h in historial],
                   marker="o", color="black", label="promedio")
    for clase in CLASES:
        ax_recall.plot(epocas, [h["val"]["recall_por_clase"][clase] for h in historial],
                       alpha=0.5, label=clase)
    ax_recall.set(title="Recall de validación por época", xlabel="Época", ylabel="Recall")
    ax_recall.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "curvas_entrenamiento.png"), dpi=120)
    plt.close(fig)


def main(argv=None):
    """Punto de entrada del entrenamiento."""
    args = parse_args(argv)
    torch.manual_seed(args.seed)  # inicialización del head y orden de los batches reproducibles

    loader_train, loader_val = preparar_dataloaders(
        args.data_dir, args.batch_size, args.seed, args.num_workers
    )
    print(f"train: {len(loader_train.dataset)} imágenes | val: {len(loader_val.dataset)} imágenes")

    model = build_model(num_classes=len(CLASES))
    criterion = nn.CrossEntropyLoss()
    # Al optimizador solo se le pasan los parámetros entrenables (el head).
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )

    historial = []
    mejor_recall = -1.0
    mejor_estado = None
    epocas_sin_mejora = 0

    for epoca in range(1, args.epochs + 1):
        inicio = time.time()
        loss_train = entrenar_una_epoca(model, loader_train, criterion, optimizer)
        metricas_val = evaluar(model, loader_val, criterion)
        segundos = time.time() - inicio

        historial.append(
            {"epoca": epoca, "loss_train": loss_train, "val": metricas_val, "segundos": segundos}
        )
        print(
            f"Época {epoca:2d} | loss train {loss_train:.4f} | loss val {metricas_val['loss']:.4f} | "
            f"recall val {metricas_val['recall_macro']:.3f} | "
            f"tumores como sanos {metricas_val['tumores_como_sanos']}/{metricas_val['tumores_total']} | "
            f"{segundos:.0f} s"
        )

        # Early stopping sobre el recall promedio de validación.
        if metricas_val["recall_macro"] > mejor_recall:
            mejor_recall = metricas_val["recall_macro"]
            mejor_estado = copy.deepcopy(model.state_dict())
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= args.patience:
                print(f"Sin mejora en {args.patience} épocas: se corta el entrenamiento.")
                break

    mejor_epoca = max(historial, key=lambda h: h["val"]["recall_macro"])["epoca"]
    print(f"Mejor época: {mejor_epoca} (recall val {mejor_recall:.3f})")

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(mejor_estado, os.path.join(args.model_dir, "model.pth"))
    guardar_curvas(historial, args.output_dir)


if __name__ == "__main__":
    main()
