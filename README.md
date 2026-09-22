# Clasificación de Tumores Cerebrales en MRI — Pipeline End-to-End en AWS

Este es mi proyecto final de la Tecnicatura Superior en Ciencias de Datos. Construyo un
pipeline completo de clasificación de imágenes de resonancia magnética cerebral (MRI)
para detección de tumores, con foco en ingeniería de datos, transfer learning e
implementación en la nube (AWS) dentro del free tier.

## Objetivo

Clasificar imágenes MRI cerebrales 2D en 4 categorías:
- **Glioma**
- **Meningioma**
- **Pituitario**
- **Sin tumor**

## Dataset

Trabajo con el [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
de Kaggle — 7.200 imágenes (1.800 por clase, balanceado), combinación de figshare +
SARTAJ + Br35H, ya dividido en Training/Testing por clase.

## Stack tecnológico

- **Prototipado (Bimestre 1):** entorno local, Python, PyTorch/TensorFlow (todavía sin definir, lo decido en la fase de modelado)
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

Documento el diseño completo de la arquitectura en AWS en [`docs/architecture.md`](docs/architecture.md).

## Documentación

| Documento | Para quién | Qué encontrás |
|---|---|---|
| [`docs/documentacion_no_tecnica.md`](docs/documentacion_no_tecnica.md) | Cualquier lector, sin conocimiento técnico (pensado para que también lo entienda alguien con formación médica) | Qué es el proyecto, cómo funciona sin tecnicismos, y sus limitaciones |
| [`docs/documentacion_tecnica.md`](docs/documentacion_tecnica.md) | Lectores técnicos | Decisiones de diseño, hallazgos del EDA y su impacto concreto en el código |
| [`docs/model_card.md`](docs/model_card.md) | Quien evalúe el modelo formalmente | Ficha técnica: arquitectura, dataset, métricas, limitaciones |
| [`docs/architecture.md`](docs/architecture.md) | Lectores técnicos | Diagrama de infraestructura en AWS y justificación de cada decisión de free tier |
| [`docs/desarrollo_asistido_por_ia.md`](docs/desarrollo_asistido_por_ia.md) | Cualquier lector interesado en la metodología de trabajo | Con qué alcance se usó IA en el desarrollo, y qué parte es criterio propio |

## Cómo correr en local

Así lo tengo armado para correr en mi máquina:

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

🚧 En desarrollo — Bimestre 1. Ya terminé el EDA (`notebooks/01_eda.ipynb`); mi próximo paso es elegir el framework y la arquitectura para arrancar el modelado.

## Autor

Franco Valentín Guerrero — Tecnicatura Superior en Ciencias de Datos
