"""
Orquestación opcional con SageMaker Pipelines (stretch goal, no
imprescindible para la entrega — ver docs/architecture.md, sección
"Imprescindible vs. opcional").

Encadenaría: preprocesamiento -> entrenamiento -> registro del modelo,
usando los mismos steps ya definidos en scripts/. No reduce costo por
sí solo, solo orquesta los pasos ya presupuestados dentro del free tier.
"""


def build_pipeline():
    """Define el pipeline de SageMaker.

    TODO: implementar solo si el tiempo del Bimestre 2 lo permite.
    """
    raise NotImplementedError
