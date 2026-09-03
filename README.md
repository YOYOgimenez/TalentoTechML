Predicción de Precios de Autos Usados — Craigslist

Proyecto de machine learning orientado a predecir el precio de venta de vehículos usados a partir de sus características técnicas y de estado, utilizando un dataset real de publicaciones de Craigslist.

Objetivo

Construir y evaluar un modelo de regresión capaz de estimar el precio de un vehículo usado, comparando un modelo lineal de referencia contra un modelo de ensamble no lineal.

Dataset

Craigslist Cars and Trucks — Kaggle

Aproximadamente 426.000 publicaciones originales. Tras la limpieza y el filtrado de outliers, el conjunto final utilizado para el modelado quedó en 262.326 registros.

Para reproducir el análisis, descargar dataset.csv y ubicarlo en la carpeta data/.

Metodología

1. Análisis exploratorio Distribución de las variables, análisis del target, detección de valores nulos y evaluación de cardinalidad de las variables categóricas.

2. Limpieza y preparación

Eliminación de columnas con más del 40% de valores faltantes y de identificadores sin valor predictivo.
Filtrado de outliers en precio, año y odómetro mediante rangos definidos por criterio de negocio.
Imputación de nulos categóricos y eliminación de registros duplicados.

3. Selección de variables Se descartó model por su alta cardinalidad (más de 29.000 valores únicos) y se conservaron las variables con mayor relación esperada con el precio: año, odómetro, tipo de combustible, transmisión, tracción, estado del vehículo, tipo de carrocería y fabricante.

4. Codificación y escalado Codificación de variables categóricas y estandarización de las numéricas.

5. Modelado y evaluación División en entrenamiento (80%) y prueba (20%). Se entrenaron dos modelos y se evaluaron con MAE, RMSE y R², complementados con análisis de residuos, importancia de variables y validación cruzada de cinco particiones.

Resultados
Modelo	MAE (USD)	RMSE (USD)	R²
Regresión Lineal	7.381	10.087	0,490
Random Forest	4.520	7.171	0,742

El modelo de Random Forest reduce el error medio absoluto en aproximadamente un 39% respecto del modelo lineal de referencia y explica cerca del 74% de la varianza del precio.

Validación cruzada (5 folds): R² de 0,739 con un desvío estándar de 0,003, lo que indica un rendimiento estable entre particiones y ausencia de dependencia respecto de una división particular de los datos.

Variables más influyentes: año del vehículo (0,547), odómetro (0,140) y tipo de combustible (0,116).

Un hallazgo relevante es que el tipo de combustible presenta una correlación lineal baja con el precio pero una importancia alta en el modelo de ensamble, lo que sugiere una relación no lineal que la regresión lineal no logra capturar.

Limitaciones
La codificación aplicada a las variables categóricas nominales introduce un orden arbitrario entre categorías, lo cual puede afectar al modelo lineal. Una codificación one-hot o basada en el target sería más adecuada.
El descarte de model simplifica el problema pero elimina una variable potencialmente muy informativa. Una agrupación por modelos más frecuentes podría recuperar parte de esa señal.
Los rangos de filtrado de outliers se definieron por criterio y no mediante un método estadístico formal.
No se realizó optimización de hiperparámetros. El Random Forest se entrenó con una configuración fija.
Próximos pasos
Aplicar transformación logarítmica al target, dada la asimetría observada en la distribución del precio.
Reemplazar la codificación actual por one-hot o target encoding.
Incorporar modelos de gradient boosting (XGBoost, LightGBM) y optimización de hiperparámetros.
Encapsular el preprocesamiento en un Pipeline para garantizar reproducibilidad sobre datos nuevos.
Stack

Python, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.

Instalación
bash
pip install -r requirements.txt