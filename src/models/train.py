"""
Script de entrenamiento del clasificador.

Diseñado para correr igual en dos entornos:
- Entorno local (Bimestre 1, prototipado)
- SageMaker Training Job (Bimestre 2, ml.m5.xlarge, CPU, on-demand, sin Spot)

Al portarlo a SageMaker, este archivo se pasa como entry_point del
Estimator; los hiperparámetros y paths de datos se leen desde
variables de entorno / argumentos (convención de SageMaker).
"""


def main():
    """Punto de entrada del entrenamiento.

    TODO: implementar en fase de modelado (Bimestre 1).
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
