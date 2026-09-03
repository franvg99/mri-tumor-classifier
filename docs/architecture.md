# Arquitectura del proyecto

## Resumen

El pipeline se divide en dos fases: prototipado en Google Colab (sin costo, sin límite de tiempo) y una fase formal en AWS acotada a la ventana de 2 meses del free tier de SageMaker.

## Diagrama de flujo

```
┌──────────────┐
│ Dataset Kaggle│
└──────┬───────┘
       ▼
┌─────────────────────┐
│   S3: raw/            │  (dentro de 5GB free tier, 12 meses)
└──────────┬───────────┘
           ▼
┌─────────────────────────────┐
│ Studio Notebook (ml.t3.medium)│  ← preprocesamiento acá,
│ resize + augmentation + split │    NO en Processing Job
└──────────┬────────────────────┘
           ▼
┌─────────────────────┐
│  S3: processed/       │
└──────────┬────────────┘
           ▼
┌───────────────────────────────┐
│ SageMaker Training Job          │
│ On-Demand ml.m5.xlarge (CPU)    │  ← dentro de 50 hs/mes,
│ EfficientNet-B0/MobileNetV2     │    SIN Spot, SIN GPU
│ (backbone congelado + head)     │
└──────────┬───────────────────────┘
           ▼
┌─────────────────────┐
│ S3: model artifacts   │──► Model Registry (metadata, sin costo de cómputo extra)
└──────────┬────────────┘
           ▼
┌─────────────────────────────┐
│ SageMaker Real-Time Endpoint   │  ← dentro de 125 hs/mes
│ ml.m5.xlarge                   │     ¡Eliminar después de probar!
└──────────┬─────────────────────┘
           ▼
   Cliente/API → {clase, probabilidad}

        ↕ CloudWatch Billing Alarm (recomendado: alerta a los $1 o $5)
```

## Decisiones de diseño y justificación

### Por qué no Spot Instances
El AWS Free Tier aplica exclusivamente a instancias On-Demand de tipos específicos (`ml.t3.medium`, `ml.m4.xlarge`/`ml.m5.xlarge`). Las Spot Instances se facturan bajo un esquema aparte y no consumen ni son cubiertas por las horas gratuitas.

### Por qué no GPU
Ninguna instancia GPU está incluida en el free tier de SageMaker. Todo el entrenamiento e inferencia corre sobre CPU (`ml.m5.xlarge`).

### Por qué preprocesamiento en notebook y no en Processing Job
Los SageMaker Processing Jobs no tienen línea de free tier propia (a diferencia de Notebooks, Training e Inferencia). Correr el preprocesamiento dentro del Studio Notebook aprovecha horas ya cubiertas por el free tier.

### Por qué un modelo liviano (EfficientNet-B0 / MobileNetV2)
Al entrenar sobre CPU dentro de las 50 horas/mes gratuitas, conviene una arquitectura con backbone congelado (transfer learning clásico) y resolución de imagen reducida, para que el fine-tuning sea viable en un tiempo razonable.

### Ventana de tiempo del free tier
El trial de SageMaker dura 2 meses desde la creación del primer recurso SageMaker (no desde la creación de la cuenta AWS). Por eso el primer recurso SageMaker se crea recién al arrancar el Bimestre 2 — la carga de datos a S3 no consume ese reloj.

## Tabla resumen del free tier relevante

| Servicio | Cubierto gratis | Instancia | Duración |
|---|---|---|---|
| Studio Notebooks | 250 horas/mes | ml.t3.medium | 2 meses |
| Entrenamiento | 50 horas/mes | ml.m4.xlarge / ml.m5.xlarge (sin GPU) | 2 meses |
| Inferencia real-time | 125 horas/mes | ml.m4.xlarge / ml.m5.xlarge | 2 meses |
| S3 | 5 GB de almacenamiento, 20.000 GET y 2.000 PUT por mes | — | 12 meses |
