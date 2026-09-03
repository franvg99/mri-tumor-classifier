"""
Lanza el SageMaker Training Job usando src/models/train.py como entry_point.

Importante (ver docs/architecture.md):
- instance_type='ml.m5.xlarge' (dentro del free tier, 50 hs/mes)
- On-Demand, NUNCA Spot (Spot no está cubierto por el free tier)
- Sin GPU (no disponible en el free tier)

Uso previsto (Bimestre 2, ya con el script de train.py validado en Colab):
    python scripts/launch_training_job.py

TODO: implementar en fase de implementación en AWS (Bimestre 2).
"""

if __name__ == "__main__":
    raise NotImplementedError
