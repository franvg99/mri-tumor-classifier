# Clasificación de Tumores Cerebrales en MRI — Pipeline End-to-End en AWS

Este es mi proyecto final de la Tecnicatura Superior en Ciencias de Datos. Construyo un
pipeline completo de clasificación de imágenes de resonancia magnética cerebral (MRI)
para detección de tumores, con foco en ingeniería de datos, transfer learning e
implementación en la nube (AWS) dentro del plan gratuito.

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

- **Prototipado (Bimestre 1):** entorno local, Python, PyTorch
- **Cloud (Bimestre 2):** AWS S3 (almacenamiento), IAM (permisos) y Amazon SageMaker (inferencia), dentro del **plan gratuito de AWS**
- **Modelo:** Transfer learning con EfficientNet-B0 (backbone preentrenado en ImageNet, congelado)
- **Entrenamiento:** Adam (`learning_rate = 0.001`), hasta 30 épocas con early stopping (paciencia 5) sobre el recall de validación

## Cronograma

| Bimestre | Período | Foco |
|---|---|---|
| 1 | Fines de agosto – fines de octubre | Documentación, arquitectura, EDA y prototipado de modelado en entorno local |
| 2 | Noviembre – 9 de diciembre | Despliegue en AWS (SageMaker), dentro del plan gratuito |

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
├── scripts/          # Puntos de entrada ejecutables (empaquetado, rol de IAM, despliegue)
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
| [`docs/architecture.md`](docs/architecture.md) | Lectores técnicos | Diagrama de infraestructura en AWS y justificación de cada decisión |
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

## Cómo desplegar en AWS

Requiere [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html),
el modelo entrenado en `models/model.pth` y un bucket propio indicado en `config/config.yaml`.
En PowerShell, con el venv activo:

```powershell
aws login --region us-east-2 --profile mri-tumor
$env:AWS_PROFILE = "mri-tumor"
python -m scripts.package_model --bucket <bucket>   # empaqueta y sube el modelo
python -m scripts.create_sagemaker_role             # solo la primera vez
python -m scripts.deploy_endpoint crear             # a partir de acá factura por hora
python -m src.inference.predictor <imagen.jpg>
python -m scripts.deploy_endpoint borrar            # siempre, al terminar
```

## Estado actual

🚧 En desarrollo — Bimestre 2. Ya terminé el EDA, el preprocesamiento, el entrenamiento y la evaluación del modelo: **92,5% de recall promedio sobre test** (1,3% de tumores predichos como "sin tumor"), excluidas las imágenes con copia casi idéntica en train, con una prueba de estrés de brillo que descarta el brillo como señal principal. Detalle en la [model card](docs/model_card.md). El modelo se desplegó en un endpoint de SageMaker el 25/11/2026: respondió igual que en local en las 40 imágenes de prueba, y se eliminó al terminar para no generar costos ([arquitectura](docs/architecture.md)). Queda pendiente el etiquetado manual de secuencias de resonancia, idealmente con un profesional médico.

## Autor

Franco Valentín Guerrero — Tecnicatura Superior en Ciencias de Datos
