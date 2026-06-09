# Proyecto Final – Programación Avanzada en Ciencia de Datos

## Clasificación de accidentes viales con víctimas fatales

Este proyecto construye un modelo predictivo en Python para estimar si una víctima de un siniestro vial corresponde a un caso **fatal** o **no fatal**, utilizando un dataset público de víctimas de siniestros viales.

El problema se trabaja como **clasificación binaria**:

- `fatal = 1` → la víctima falleció.
- `fatal = 0` → la víctima no falleció.

La consigna menciona “probabilidad de que un accidente tenga víctimas fatales”. Por eso, además de predecir la clase final, el proyecto guarda la columna `probabilidad_fatal`, generada con `predict_proba()` del modelo entrenado.

> Aclaración técnica: aunque la consigna menciona “algoritmos de regresión”, las métricas solicitadas —Accuracy, Precision, Recall y F1-score— corresponden a un problema de clasificación. Por ese motivo se comparan modelos de clasificación supervisada.

---

## Objetivo

Construir un flujo reproducible de ciencia de datos que incluya:

- Limpieza y preparación del dataset.
- Creación de una variable objetivo binaria.
- Pipeline de preprocesamiento con `scikit-learn`.
- Comparación de modelos.
- Validación cruzada estratificada.
- Ajuste de hiperparámetros con `GridSearchCV`.
- Evaluación final sobre conjunto de prueba.
- Almacenamiento de datos y resultados en MongoDB.
- Generación de gráficos y archivos de salida.

---

## Metodología

### 1. Preparación de datos

El script `src/preprocessing.py` realiza las siguientes tareas:

- Carga del archivo original `siniestros_viales_victimas.xlsx`.
- Reemplazo de valores `SD` por valores nulos.
- Conversión de columnas de fecha.
- Creación de la variable objetivo `fatal` a partir de `fecha_fallecimiento_victima`.
- Generación de variables temporales desde `fecha_siniestro`:
  - `anio`
  - `mes`
  - `dia_semana`
- Conversión de `edad_victima` a formato numérico.
- Exportación del dataset limpio en `data/processed/dataset_limpio.csv`.

---

### 2. Variables utilizadas

**Variables predictoras:**

| Variable | Tipo | Descripción |
|---|---|---|
| `anio` | Numérica | Año del siniestro |
| `edad_victima` | Numérica | Edad de la víctima |
| `mes` | Numérica | Mes del siniestro |
| `dia_semana` | Numérica | Día de la semana del siniestro |
| `sexo_victima` | Categórica | Sexo de la víctima |
| `modo_desplazamiento_victima` | Categórica | Forma de desplazamiento |
| `rol_victima` | Categórica | Rol de la víctima en el siniestro |

**Variable objetivo:**

| Variable | Descripción |
|---|---|
| `fatal` | 1 si la víctima falleció, 0 si no falleció |

---

### 3. División train/test

El dataset se divide en:

- **80% entrenamiento**
- **20% prueba**

Se utiliza `stratify=y` para conservar la proporción de casos fatales y no fatales tanto en entrenamiento como en prueba.

El conjunto de prueba se mantiene separado durante el entrenamiento y la búsqueda de hiperparámetros. Solo se utiliza al final para medir el desempeño real del modelo sobre datos no vistos.

---

### 4. Pipeline de preprocesamiento

El entrenamiento usa un `Pipeline` de `scikit-learn` con un `ColumnTransformer`.

Para variables numéricas:

- Imputación con la media.
- Escalado con `StandardScaler`.

Para variables categóricas:

- Imputación con la moda.
- Codificación `OneHotEncoder`.

Esto evita fuga de información, porque las transformaciones se ajustan solo con los datos de entrenamiento y luego se aplican al conjunto de prueba.

---

## Modelos comparados

Se comparan dos algoritmos de clasificación:

1. **Logistic Regression**
2. **Random Forest Classifier**

Ambos se configuran con `class_weight="balanced"` para compensar el fuerte desbalance entre clases.

---

## Validación cruzada y ajuste de hiperparámetros

El proyecto utiliza `GridSearchCV` con **validación cruzada estratificada de 5 particiones** mediante `StratifiedKFold`.

La validación cruzada se aplica únicamente sobre el conjunto de entrenamiento. El conjunto de prueba queda reservado para la evaluación final.

Configuración aplicada:

- Método: `StratifiedKFold`
- Cantidad de particiones: `5`
- Métricas calculadas en CV:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Métrica principal para seleccionar el mejor modelo: `F1-score`

Se eligió F1-score como métrica principal porque el dataset está desbalanceado y esta métrica permite equilibrar la precisión del modelo con su capacidad para detectar casos fatales.

Los resultados de validación cruzada se guardan en:

```bash
outputs/metrics/cv_results.csv
```

---

## Desbalance de clases

El dataset presenta un fuerte desbalance:

- Clase no fatal (`0`): aproximadamente 61.000 casos.
- Clase fatal (`1`): aproximadamente 600 casos.

Esto significa que Accuracy puede ser engañosa: un modelo podría obtener una exactitud alta prediciendo casi siempre la clase mayoritaria.

Por eso, además de Accuracy, se analizan:

- **Precision:** qué tan confiables son las predicciones de fatalidad.
- **Recall:** qué proporción de casos fatales logra detectar el modelo.
- **F1-score:** equilibrio entre Precision y Recall.

---

## Resultados obtenidos

### Evaluación final sobre test

Resultados obtenidos con el modelo final:

| Métrica | Valor |
|---|---:|
| Accuracy | 0.9946 |
| Precision | 0.8767 |
| Recall | 0.5246 |
| F1-score | 0.6564 |

### Matriz de confusión

|  | Predicción no fatal | Predicción fatal |
|---|---:|---:|
| Real no fatal | 12285 | 9 |
| Real fatal | 58 | 64 |

### Interpretación

El modelo logra una Accuracy muy alta, pero esta métrica debe interpretarse con cuidado por el fuerte desbalance de clases.

La Precision de 0.8767 indica que, cuando el modelo predice fatalidad, suele hacerlo correctamente. El Recall de 0.5246 muestra que detecta aproximadamente la mitad de los casos fatales reales. Esto evidencia que todavía existen falsos negativos, es decir, casos fatales que el modelo clasifica como no fatales.

El F1-score de 0.6564 representa un equilibrio moderado entre precisión y detección de casos fatales.

---

## Visualizaciones generadas

El script `src/visualize.py` genera los siguientes gráficos:

| Archivo | Descripción |
|---|---|
| `outputs/desbalance_clases.png` | Distribución de casos fatales y no fatales |
| `outputs/metricas_modelo.png` | Métricas finales del modelo en test |
| `outputs/matriz_confusion.png` | Matriz de confusión |
| `outputs/validacion_cruzada_f1.png` | Mejores resultados de validación cruzada según F1-score |
| `outputs/importancia_variables.png` | Importancia de variables del Random Forest |

---

## Base de datos: MongoDB

El proyecto usa MongoDB para almacenar datos de entrada, configuración del modelo y resultados.

Base de datos:

```text
accidentes_viales
```

Colecciones:

| Colección | Contenido |
|---|---|
| `datos_entrada` | Dataset limpio y procesado |
| `configuracion_modelo` | Variables, modelo ganador, hiperparámetros y validación cruzada |
| `resultados_modelo` | Métricas finales, matriz de confusión y archivo de predicciones |

El proyecto está preparado para no fallar si MongoDB no está iniciado. En ese caso, muestra un aviso y continúa generando los archivos locales.

Para guardar los resultados en MongoDB, iniciar el servicio antes de ejecutar el pipeline:

```bash
mongod
```

---

## Estructura del proyecto

```text
proyecto-final-reformado/
│
├── data/
│   ├── raw/
│   │   └── siniestros_viales_victimas.xlsx
│   └── processed/
│       └── dataset_limpio.csv
│
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── exploracion.ipynb
│
├── outputs/
│   ├── X_test.csv
│   ├── y_test.csv
│   ├── predicciones_test.csv
│   ├── desbalance_clases.png
│   ├── metricas_modelo.png
│   ├── matriz_confusion.png
│   ├── validacion_cruzada_f1.png
│   ├── importancia_variables.png
│   └── metrics/
│       ├── metricas.json
│       └── cv_results.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── evaluate.py
│   ├── plots.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── utils.py
│   └── visualize.py
│
├── .env.example
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

El archivo `src/data.py` fue eliminado porque duplicaba lógica de carga/preprocesamiento y podía generar confusión con el flujo principal.

---

## Instalación

### 1. Crear entorno virtual

```bash
python -m venv venv
```

### 2. Activar entorno virtual

En Windows:

```bash
venv\Scripts\activate
```

En Linux/Mac:

```bash
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Ejecución del proyecto

Se puede ejecutar el flujo completo con:

```bash
python main.py
```

También se puede ejecutar paso a paso:

### 1. Preprocesamiento

```bash
python -m src.preprocessing
```

### 2. Entrenamiento, GridSearchCV y validación cruzada

```bash
python -m src.train
```

### 3. Evaluación final sobre test

```bash
python -m src.evaluate
```

### 4. Visualizaciones

```bash
python -m src.visualize
```

---

## Verificación en MongoDB

Si MongoDB está iniciado, se pueden consultar las colecciones con:

```javascript
use accidentes_viales

db.datos_entrada.find().limit(5)
db.configuracion_modelo.find()
db.resultados_modelo.find()
```

---

## Archivos de salida principales

| Archivo | Uso |
|---|---|
| `models/best_model.pkl` | Modelo final entrenado |
| `outputs/metrics/cv_results.csv` | Resultados de validación cruzada |
| `outputs/metrics/metricas.json` | Métricas finales sobre test |
| `outputs/predicciones_test.csv` | Predicciones finales y probabilidad de fatalidad |
| `outputs/*.png` | Gráficos para informe o presentación |

---

## Limitaciones y mejoras futuras

El modelo actual permite construir una primera solución predictiva, pero todavía tiene limitaciones:

- El Recall de la clase fatal indica que el modelo no detecta todos los casos fatales.
- El dataset está fuertemente desbalanceado.
- Se usan pocas variables explicativas.
- No se realizó ajuste del umbral de clasificación.

Posibles mejoras:

- Incorporar más variables del siniestro.
- Ajustar el umbral de clasificación para priorizar Recall.
- Probar técnicas específicas para desbalance, como SMOTE o submuestreo.
- Comparar más modelos.
- Analizar curvas ROC y Precision-Recall.
- Generar un dashboard en Power BI, Plotly o Streamlit.

---

## Tecnologías utilizadas

- Python
- Pandas
- NumPy
- Scikit-learn
- MongoDB
- PyMongo
- Matplotlib
- Joblib
- OpenPyXL

---

## Estado del proyecto

- Preprocesamiento completo.
- Pipeline de modelado implementado.
- Comparación de modelos realizada.
- Validación cruzada estratificada agregada.
- Resultados de CV exportados a CSV.
- Evaluación final sobre test implementada.
- Predicción de probabilidad de fatalidad agregada.
- Visualizaciones generadas.
- MongoDB integrado con manejo de error si el servicio no está activo.
- `src/data.py` eliminado por estar de más.

---


## Autores

 Moyano Leonardo, 
 Diego German Benitez Tolomei, 
 Diego Labbozzetta
 
Licenciatura en Ciencia de Datos – Programación Avanzada (2026)

---
