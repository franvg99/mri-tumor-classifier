[⬅ Volver al README](../README.md)

# Documentación no técnica

> Este documento explica el proyecto sin dar por sentado conocimiento de programación ni
> de machine learning. Si buscás el detalle técnico (arquitectura, código, infraestructura),
> ver [`documentacion_tecnica.md`](documentacion_tecnica.md).

## Qué es este proyecto

Es mi trabajo final de la Tecnicatura Superior en Ciencias de Datos. Construyo un
**clasificador automático de imágenes de resonancia magnética (MRI) cerebral**. Dada una
imagen de un corte del cerebro, el sistema intenta decir a cuál de estas 4 categorías
pertenece:

- **Glioma**
- **Meningioma**
- **Tumor pituitario**
- **Sin tumor**

## Qué no es este proyecto

**No es una herramienta de diagnóstico clínico.** Es un proyecto académico, que entreno y
evalúo sobre un dataset público de investigación, sin validación regulatoria ni revisión
por profesionales de la salud sobre su desempeño real. No lo pienso como un reemplazo ni
una asistencia al criterio de un radiólogo o médico en la práctica. Mi objetivo es
demostrar un proceso completo de trabajo con datos e inteligencia artificial, de
principio a fin.

## Cómo funciona el modelo, explicado sin tecnicismos

El sistema no "razona" como lo haría un profesional de la salud. Es un modelo estadístico
que entreno para que reconozca patrones visuales asociados a cada categoría después de
analizar miles de ejemplos ya etiquetados (7.200 imágenes, 1.800 de cada clase). En vez
de entrenarlo por completo desde cero, reutilizo una red que ya sabe reconocer formas y
texturas en imágenes en general (entrenada previamente sobre millones de fotografías), y
solo le enseño la parte final: traducir esos patrones visuales a una de las 4 categorías
de este proyecto. Esta técnica se conoce como *transfer learning* y la elijo porque es
habitual en proyectos con recursos de cómputo limitados como este.

## Por qué la métrica prioritaria no es la cantidad total de aciertos

En un contexto sanitario, no todos los errores tienen el mismo costo. Si el modelo indica
"sin tumor" cuando en realidad había un tumor (**falso negativo**), la consecuencia es
mucho más grave que si indica "hay un tumor" cuando no lo había (**falso positivo**, que
en la práctica se traduce en un estudio adicional, no en un daño). Por eso no optimizo la
"exactitud" (accuracy, el porcentaje total de aciertos), sino el **recall o
sensibilidad**: la capacidad de no dejar pasar casos positivos. Un modelo puede tener
accuracy alta y aun así fallar sistemáticamente en la clase que menos margen de error
admite — el recall es la métrica que expone esa falla.

## Limitaciones conocidas hasta ahora

Las encontré durante el análisis exploratorio de los datos (EDA), antes incluso de
entrenar el modelo:

- **El dataset combina 3 fuentes distintas de imágenes**, y la clase "Sin tumor" viene
  enteramente de una fuente diferente a las tres clases con tumor. Detecté diferencias
  sistemáticas de brillo y contraste entre clases que coinciden con esa mezcla de
  fuentes — existe el riesgo de que el modelo aprenda a reconocer "de qué máquina/fuente
  vino la imagen" en vez de reconocer la patología real. Tomo medidas para mitigarlo
  (variar artificialmente el brillo/contraste durante el entrenamiento), pero es una
  limitación estructural del dataset que no puedo eliminar del todo.
- **Las imágenes no vienen etiquetadas con el tipo de corte** (axial, sagital o coronal),
  y el dataset mezcla los tres sin indicarlo.
- Los resultados de desempeño real (qué tan bien clasifica) todavía no existen — el
  modelo no está entrenado. Actualizo este documento cuando tenga métricas concretas.

## Estado actual del proyecto

Ya terminé el análisis exploratorio de datos (EDA). Todavía no entrené el modelo — mi
próximo paso es elegir la arquitectura y el framework a usar. Ver
[`model_card.md`](model_card.md) para el detalle técnico-formal del modelo (que hoy tiene
varios campos pendientes, justamente porque el modelo todavía no existe).

---

**Ver también:** [documentación técnica](documentacion_tecnica.md) ·
[model card](model_card.md) · [arquitectura en AWS](architecture.md) ·
[desarrollo asistido por IA](desarrollo_asistido_por_ia.md)
