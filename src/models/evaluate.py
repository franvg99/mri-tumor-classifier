"""
Evaluación del modelo.

Métricas priorizadas por el contexto sanitario del problema (un falso
negativo — no detectar un tumor real — es más costoso que un falso
positivo):

- Recall / Sensibilidad (por clase)
- F1-Score (por clase y macro)
- Matriz de confusión
- ROC-AUC (one-vs-rest, multiclase)
"""


def compute_metrics(y_true, y_pred, y_proba=None):
    """Calcula recall, F1, matriz de confusión y ROC-AUC.

    TODO: implementar en fase de modelado (Bimestre 1), usando scikit-learn.
    """
    raise NotImplementedError


def plot_confusion_matrix(y_true, y_pred, class_names):
    """Genera la matriz de confusión como imagen, para incluir en la documentación final.

    TODO: implementar en fase de modelado (Bimestre 1).
    """
    raise NotImplementedError
