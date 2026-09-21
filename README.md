# Clasificación de Tumores Cerebrales en MRI — Pipeline End-to-End en AWS

Proyecto final de la Tecnicatura Superior en Ciencias de Datos. Pipeline completo de clasificación de imágenes de resonancia magnética cerebral (MRI) para detección de tumores, con foco en ingeniería de datos, transfer learning e implementación en la nube (AWS) dentro del free tier.

## Objetivo

Clasificar imágenes MRI cerebrales 2D en 4 categorías:
- **Glioma**
- **Meningioma**
- **Pituitario**
- **Sin tumor**

## Dataset

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Kaggle) — 7.023 imágenes, combinación de figshare + SARTAJ + Br35H, ya dividido en Training/Testing por clase.

## Stack tecnológico

- **Prototipado (Bimestre 1):** entorno local, Python, PyTorch/TensorFlow (a definir en fase de modelado)
- **Cloud (Bimestre 2):** AWS S3 (almacenamiento), Amazon SageMaker (entrenamiento e inferencia), dentro del **AWS Free Tier**
- **Modelo:** Transfer learning con arquitectura liviana (EfficientNet-B0 / MobileNetV2 — a confirmar en fase de modelado)

## Cronograma

| Bimestre | Período | Foco |
|---|---|---|
| 1 | Fines de agosto – fines de octubre | Documentación, arquitectura, EDA y prototipado de modelado en entorno local |
| 2 | Noviembre | Modelado e implementación en AWS (SageMaker), dentro del free tier |

## Estructura del repositorio

```
mri-tumor-classifier/
├── config/           # Configuración (hiperparámetros, paths S3)
├── notebooks/        # EDA y prototipado exploratorio
├── src/
│   ├── data/          # Preprocesamiento, augmentation, utilidades S3
│   ├── models/        # Arquitectura, entrenamiento, evaluación
│   ├── inference/      # Handlers de inferencia (SageMaker endpoint)
│   └── pipeline/       # Orquestación (SageMaker Pipelines, opcional)
├── scripts/          # Puntos de entrada ejecutables (upload, training job, deploy)
├── tests/            # Tests unitarios
└── docs/             # Documentación técnica (arquitectura, model card)
```

Ver [`docs/architecture.md`](docs/architecture.md) para el diseño completo de la arquitectura en AWS.

## Cómo correr en local

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -r requirements.txt
python -m ipykernel install --user --name mri-tumor-classifier --display-name "mri-tumor-classifier (.venv)"
```

Creá un `.env` en la raíz con tu token de Kaggle (no se versiona):

```
KAGGLE_API_TOKEN=tu_token
```

Después abrí `notebooks/01_eda.ipynb` en VS Code y seleccioná el kernel `mri-tumor-classifier (.venv)`.

## Estado actual

🚧 En desarrollo — Bimestre 1 (documentación y prototipado).

## Autor

Franco Valentín Guerrero — Tecnicatura Superior en Ciencias de Datos
