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


def brightness_stress_test(model, df_test, factores=(0.6, 0.8, 1.0, 1.2, 1.4)):
    """Mide si el modelo usa el brillo como atajo para predecir la clase.

    Multiplica el brillo de las imágenes de test con tumor por cada factor y
    reporta qué fracción pasa a predecirse como "notumor". Si esa fracción crece
    al aclarar las imágenes, el modelo se apoya en el brillo (señal de la fuente
    Br35H, no de la patología).

    Motivación: un clasificador que solo ve brillo medio y desvío acierta el 63%
    (azar: 25%), y como test proviene de las mismas fuentes que train, la métrica
    de test no puede detectar el atajo. Ver docs/documentacion_tecnica.md,
    sección 4 de los hallazgos del EDA.

    TODO: implementar en fase de evaluación (Bimestre 1).
    """
    raise NotImplementedError
