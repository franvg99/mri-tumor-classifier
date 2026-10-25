# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Estado del repositorio

**Ya están implementados `src/data/`** (`eda.py`, `dataset.py`, `preprocessing.py`, `augmentation.py`, con tests), **`src/models/architecture.py`** (con tests) y **`src/models/train.py`** (sin tests todavía). El modelo entrenado queda en `models/model.pth` junto con `historial.json` y `curvas_entrenamiento.png` (no versionados). **El resto de `src/` sigue siendo un esqueleto:** sus funciones y todos los entry points de `scripts/` levantan `NotImplementedError`, y sus tests son `pytest.skip()`. Sus docstrings NO describen comportamiento existente: describen el contrato que hay que implementar y en qué fase corresponde hacerlo. Al trabajar en esos stubs, la tarea típica es completarlos respetando su docstring, no refactorizar código que ya anda.

`notebooks/01_eda.ipynb` ya tiene el EDA armado y anda de punta a punta en local. `02_preprocessing_debug.ipynb` verifica el split y el augmentation y reproduce los experimentos de los atajos del dataset (brillo y secuencias). `03_model_experiments.ipynb` sigue vacío, con solo un título en markdown.

El repo ya está inicializado (`.git/` existe) con remoto en GitHub (`franvg99/mri-tumor-classifier`).

## Comandos

```bash
python -m venv .venv
.venv/Scripts/activate                        # Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt

pytest                                        # todos los tests
pytest tests/test_model.py                    # un archivo
pytest tests/test_model.py::test_build_model_output_shape_placeholder  # un test
```

No hay linter, formateador ni Makefile configurados en el proyecto. `pytest.ini` agrega la raíz al `sys.path` (`pythonpath = .`) para que los tests importen `src`; sin él, `pytest` a secas falla en la recolección.

`requirements.txt` incluye `torch` y `torchvision` (framework ya decidido: PyTorch). No agregues `tensorflow`.

### Entorno local (Windows)

El proyecto se desarrolla en Windows dentro de una carpeta OneDrive, cuya ruta ya es larga de por sí — sumada a la de `.venv/site-packages`, algunos paquetes (`sagemaker`, `jedi`) superan el límite de 260 caracteres de Windows. Por eso hace falta tener habilitado `LongPathsEnabled=1` en `HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem` (cambio de registro a nivel de sistema, no del proyecto) antes de instalar `requirements.txt` completo.

Para correr los notebooks localmente en VS Code, el venv se registra como kernel de Jupyter:

```bash
python -m ipykernel install --user --name mri-tumor-classifier --display-name "mri-tumor-classifier (.venv)"
```

Después hay que seleccionarlo a mano en el selector de kernel de VS Code — no se detecta solo.

### Credenciales de Kaggle

El EDA descarga el dataset con `kagglehub`, que necesita un token de Kaggle. Vive en un `.env` en la raíz del proyecto (no versionado, bloqueado en `.gitignore`):

```
KAGGLE_API_TOKEN=...
```

Kaggle dejó de usar el par `username`/`key` del `kaggle.json` viejo; ahora es un token único. El notebook lo carga con `python-dotenv` (`load_dotenv()`) al arrancar.

## Decisiones pendientes (no asumirlas)

`config/config.yaml` marca con `TBD` lo que todavía no se decidió, y varios stubs dependen de eso:

- **`s3_bucket`** y confirmación de `region` — sin definir.

**Ya resuelto:**
- **Split 70/10/20** (19/10/2026), estratificado, deduplicado antes, `seed=42` — test agrandado para acotar el margen de error del recall por clase. `split_dataset(df)` recibe el DataFrame de `eda.listar_imagenes()` y devuelve 3 DataFrames.
- **Sin balanceo de clases** tras deduplicar (ratio 1.06; justificado en `docs/documentacion_tecnica.md`, punto 1).
- **Augmentation:** `RandomAffine(±10°, zoom ±10%)` + `ColorJitter(brillo/contraste 0.4)`, sin flip.
- `image_size` = 224, decidido en el EDA (sección 3 de `01_eda.ipynb`) para alinear con la resolución preentrenada de EfficientNet-B0/MobileNetV2 sobre ImageNet.
- **Framework: PyTorch** (27/9/2026) — sobre TensorFlow, priorizando la vigencia de PyTorch en el mercado laboral (investigación, LLMs/Hugging Face) por sobre la curva de entrada más suave de Keras.
- **Arquitectura: EfficientNet-B0** (27/9/2026) — sobre MobileNetV2. Con el backbone congelado no hay backward pass a través de él, así que la diferencia de costo entre ambas es irrelevante acá; en cambio la mejor accuracy de transferencia de EfficientNet-B0 pesa más dado que la métrica prioritaria es recall (contexto clínico).

- **Entrenamiento** (25/10/2026): Adam, `learning_rate=0.001`, techo de 30 épocas con early stopping (paciencia 5) sobre el **recall macro de validación**, no la loss. La corrida de referencia cortó en la época 20, la mejor fue la 15 (recall val 92,8%; 3/533 tumores clasificados como notumor). El backbone se mantiene en `eval()` durante el entrenamiento para que sus BatchNorm no actualicen estadísticas (ver `train.modo_entrenamiento`).

Todo actualizado en `config/config.yaml` y `docs/model_card.md`.

Al resolver alguna de las pendientes, actualizá `config/config.yaml`, `docs/model_card.md` y la sección de stack del `README.md`, que hoy las listan como TBD en paralelo.

## Arquitectura

Clasificación multiclase (4 clases: glioma, meningioma, pituitary, notumor) de MRI cerebral 2D, con transfer learning y backbone congelado. Ver `docs/architecture.md` para el diagrama de flujo completo y la justificación de cada decisión de infra.

### El proyecto tiene dos fases, y eso condiciona el código

- **Bimestre 1 — prototipado en entorno local** (venv del proyecto, sin reloj corriendo): EDA, preprocesamiento, augmentation, modelado y evaluación.
- **Bimestre 2 — AWS SageMaker** dentro del free tier: entrenamiento, registro y endpoint.

La regla de diseño central es que **el mismo código corre en los dos entornos sin modificarse**: `src/data/preprocessing.py` se porta tal cual del entorno local al Studio Notebook, y `src/models/train.py` se pasa como `entry_point` del Estimator de SageMaker. Por eso `train.py` debe leer hiperparámetros y paths de datos desde variables de entorno / argumentos CLI (convención SageMaker), nunca hardcodeados ni tomados de un notebook.

### Convenciones de SageMaker que hay que respetar

- `src/inference/inference.py` implementa las 4 funciones que SageMaker espera en script mode: `model_fn` (carga desde `model_dir`), `input_fn` (deserializa la imagen), `predict_fn`, `output_fn` (serializa a `{clase, probabilidad}`). Los nombres y firmas son fijos, los impone SageMaker.
- `src/inference/predictor.py` es el cliente para invocar el endpoint ya desplegado; no comparte código con `inference.py`.
- `src/pipeline/sagemaker_pipeline.py` es un stretch goal opcional. No lo priorices.

### Restricciones duras del AWS Free Tier

Estas no son preferencias, son las que hacen que el proyecto no genere costo. Si escribís código de infra, respetalas:

- **On-Demand siempre, Spot nunca.** Las Spot Instances no están cubiertas por el free tier.
- **Sin GPU.** Todo entrena e infiere en `ml.m5.xlarge` (CPU). De ahí que la arquitectura sea liviana y el backbone vaya congelado.
- **El preprocesamiento va en el Studio Notebook, no en un Processing Job.** Los Processing Jobs no tienen línea de free tier propia; las horas de notebook sí.
- **El endpoint se borra apenas se termina de probar** (`delete_endpoint`) — factura por hora aunque no reciba tráfico.
- **El reloj de 2 meses del trial de SageMaker arranca con el primer recurso SageMaker creado**, no con la cuenta AWS. Subir datos a S3 no lo dispara; crear un notebook sí. No sugieras crear recursos SageMaker durante la fase de ingesta.

Presupuesto: 250 hs/mes notebooks, 50 hs/mes entrenamiento, 125 hs/mes inferencia (2 meses); S3 5 GB (12 meses).

## Criterios del dominio

- **Métrica prioritaria: recall/sensibilidad**, no accuracy. Es un contexto sanitario: un falso negativo (tumor no detectado) es más costoso que un falso positivo. `src/models/evaluate.py` debe reportar recall y F1 por clase, matriz de confusión y ROC-AUC one-vs-rest.
- **El augmentation es deliberadamente conservador** (rotaciones ±10-15°, zoom leve, brillo/contraste). Evitar deformaciones agresivas que distorsionen features clínicamente relevantes.
  - **Sin flip horizontal:** el EDA (notebook `01_eda.ipynb`, sección 4) encontró que el dataset mezcla cortes axial/sagital/coronal sin etiquetar, y el flip es anatómicamente inválido en sagital.
  - **Brillo/contraste no opcional:** el EDA (sección 5) encontró diferencias sistemáticas de intensidad entre clases que coinciden con la composición del dataset por fuente (notumor=100% Br35H) — riesgo de que el modelo aprenda la fuente en vez de la patología.
- El augmentation se aplica **solo a train**; val/test usan el pipeline de `get_eval_pipeline()` (resize + normalize nada más). Es la única definición del preprocesamiento de evaluación: `inference.py` debe importarla, no reimplementarla.
- **Atajos medidos en el dataset** (ver `docs/documentacion_tecnica.md`, puntos 4 y 6): brillo/desvío solos predicen la clase con 63% de accuracy, y las clases con tumor son casi todas T1 con contraste mientras notumor mezcla T1/T2/FLAIR. Ninguno se corrige con preprocesamiento; `evaluate.py` debe medirlos (prueba de estrés de brillo, recall por secuencia sobre una muestra etiquetada a mano). El uso previsto está acotado a T1 con contraste.
- El split debe ser **estratificado por clase** y con `seed` fijo.

## Datos

El dataset ([Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), 7.200 imágenes — 1.800 por clase, balanceado) **no se versiona**: vive en S3 bajo los prefijos `raw/`, `processed/` y `models/`. El `.gitignore` bloquea `data/`, `models/` y todas las extensiones de imagen y de pesos (`.pt`, `.pth`, `.h5`, `.ckpt`, `.npy`). No agregues datos ni artefactos de modelo al repo.

## Idioma

Toda la documentación, los docstrings y los comentarios están en español rioplatense. Mantené ese registro al escribir código o docs nuevos.
