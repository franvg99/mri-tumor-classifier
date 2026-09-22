[⬅ Volver al README](../README.md)

# Desarrollo asistido por IA

> Este documento describe con qué alcance usé inteligencia artificial (Claude, de
> Anthropic) durante el desarrollo de este proyecto, y qué parte del trabajo corresponde
> a mi propio criterio y decisiones. Lo escribo con fines de transparencia académica.

## Por qué escribo este documento

Buena parte del código, la documentación y el análisis exploratorio de este repositorio
los construí con asistencia de un modelo de lenguaje (Claude Code). Es información
relevante para evaluar el trabajo, así que la documento de forma explícita en vez de
darla por sobreentendida.

La idea central: **la IA me acelera tareas de ejecución, no reemplaza mi criterio.** Cada
decisión de diseño del proyecto (arquitectura, hiperparámetros, interpretación de
resultados) la tomé y validé yo, no la delegué.

## Qué tareas me acelera la IA

- **Scaffolding y configuración de entorno:** estructura inicial del repositorio, setup
  del entorno virtual local, resolución de problemas de entorno específicos de Windows
  (límite de rutas largas, registro del kernel de Jupyter).
- **Escritura de código repetitivo o mecánico:** helpers de EDA (`src/data/eda.py`),
  celdas de notebook siguiendo un patrón que ya acordé, correcciones de bugs puntuales
  (por ejemplo, un color ilegible en un gráfico, una celda de notebook corrompida por una
  edición concurrente).
- **Redacción de documentación:** primer borrador de los documentos en `docs/`, a partir
  de decisiones y hallazgos que ya discutí y confirmé en la conversación — no generados
  de forma autónoma sin contexto del proyecto.
- **Búsqueda de información puntual:** por ejemplo, confirmar la composición por fuente
  del dataset de Kaggle (qué clase proviene de qué dataset original), que después resultó
  relevante para una decisión de diseño (ver más abajo).
- **Ejecución y lectura de resultados:** correr celdas, generar gráficos comparativos,
  resumir salidas numéricas para agilizar la discusión.

## Qué parte fue trabajo y criterio propio

Ejemplos concretos de esta misma colaboración, no afirmaciones genéricas:

- **Corrección de un razonamiento incompleto.** Ante una recomendación inicial de
  `image_size = 150` (justificada solo por minimizar el upscaling), noté que ese criterio
  ignoraba que la mayoría del dataset son imágenes de 512×512, y que reducir a 150
  descartaba más del 90% de su resolución real sin necesidad. Esa observación llevó a
  rehacer el análisis y cambiar la recomendación final a 224.
- **Verificación de datos en vez de aceptarlos.** Ante la cifra de "7.023 imágenes" que ya
  figuraba en la documentación heredada del scaffolding inicial, la contrasté con lo que
  muestra la página de Kaggle y detecté la discrepancia con las 7.200 imágenes reales del
  dataset descargado, lo que llevó a corregir la documentación en varios archivos.
- **Todas las decisiones de diseño abiertas** (`image_size`, exclusión del flip
  horizontal en el augmentation, migración del flujo de trabajo de Google Colab a un
  entorno local, estructura y tono de la documentación) me las presentaron como
  disyuntivas con sus argumentos a favor y en contra, y las decidí yo — no las asumió la
  IA.
- **Ritmo de trabajo que dirigí yo:** hice el desarrollo pidiendo explicaciones de cada
  paso antes de avanzar, en tramos chicos y revisables, en vez de aceptar cambios grandes
  de una sola vez.

## Qué significa esto en términos prácticos

- El código de `src/data/eda.py` lo escribí con asistencia de IA, pero cada resultado que
  produce lo revisé e interpreté yo antes de tomarlo como base para una decisión.
- Las conclusiones del EDA (`notebooks/01_eda.ipynb`) reflejan lecturas y decisiones que
  discutí y confirmé explícitamente, no generadas de forma automática a partir de los
  números.
- Los stubs pendientes de `src/` (marcados con `NotImplementedError`) los voy a
  implementar con el mismo esquema: asistencia en la escritura, criterio propio en las
  decisiones.

---

**Ver también:** [documentación técnica](documentacion_tecnica.md) ·
[documentación no técnica](documentacion_no_tecnica.md)
