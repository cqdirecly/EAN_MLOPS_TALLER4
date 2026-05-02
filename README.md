# Pipeline CI/CD para Machine Learning con MLflow

Pipeline automatizado de **entrenamiento, registro y validación** de un modelo de Machine Learning utilizando **MLflow** y **GitHub Actions**. El proyecto implementa un flujo CI/CD completo donde, ante cada push a la rama `main`, se ejecuta un pipeline que entrena un modelo de regresión, lo registra con MLflow, lo valida contra umbrales de calidad y publica los artefactos resultantes.

---

## Tabla de contenidos

1. [Objetivo del proyecto](#objetivo-del-proyecto)
2. [Arquitectura del pipeline](#arquitectura-del-pipeline)
3. [Dataset utilizado](#dataset-utilizado)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Requisitos](#requisitos)
6. [Instalación y uso local](#instalación-y-uso-local)
7. [Pipeline de CI/CD](#pipeline-de-cicd)
8. [Detalle de componentes](#detalle-de-componentes)
9. [Métricas y umbrales de validación](#métricas-y-umbrales-de-validación)
10. [Trazabilidad y auditoría](#trazabilidad-y-auditoría)

---

## Objetivo del proyecto

Construir un pipeline reproducible que automatice las siguientes etapas en cada actualización del repositorio:

- **Entrenamiento** de un modelo de regresión sobre el dataset Wine Quality.
- **Registro** del modelo, parámetros, métricas, signature e input_example en MLflow.
- **Validación** del modelo contra umbrales de calidad predefinidos usando datos externos.
- **Publicación** de artefactos auditables (modelo serializado y tracking completo de MLflow) para revisión y reproducibilidad.

---

## Arquitectura del pipeline

```
                  push a main
                      |
                      v
            +---------------------+
            |   GitHub Actions    |
            +---------------------+
                      |
        +-------------+-------------+
        |                           |
        v                           v
+----------------+         +-----------------+
|  make train    |         |  make validate  |
|  (train.py)    |  ---->  |  (validate.py)  |
+----------------+         +-----------------+
        |                           |
        | registra en MLflow        | carga modelo de MLflow
        | con signature e           | y valida con datos
        | input_example             | externos
        v                           v
+-------------------------------------------+
|     mlruns/  +  model.pkl  +  artifacts   |
+-------------------------------------------+
                      |
                      v
        Publicación de artefactos en GitHub
```

---

## Dataset utilizado

### Wine Quality (Red Wine) — UCI Machine Learning Repository

Se utiliza el dataset **Wine Quality - Red Wine** del repositorio público UCI Machine Learning Repository.

- **Fuente:** https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
- **Tipo de problema:** Regresión (predecir calidad del vino).
- **Muestras:** 1.599 vinos tintos portugueses ("Vinho Verde").
- **Features (11):** fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol.
- **Variable objetivo:** `quality` (calificación entera entre 3 y 8 asignada por catadores).

### Justificación de la elección

1. **Es un dataset externo y público,** no incluido en `sklearn.datasets`, lo que cumple el requisito del taller de no usar datasets embebidos en librerías.
2. **Es el dataset oficial usado por MLflow en su tutorial de quickstart,** lo que garantiza compatibilidad y precedente comprobado.
3. **Tiene un balance ideal para el aprendizaje:** ni muy simple (Iris) ni muy complejo (datasets de imágenes); permite mostrar buenas prácticas sin saturar el pipeline.
4. **Permite explorar regresión con features físicas interpretables,** facilitando explicaciones del modelo.
5. **Es lo suficientemente pequeño** (~85 KB) para incluirlo en el repositorio y garantizar reproducibilidad sin dependencias externas durante el CI.

### Procesamiento aplicado

- Lectura del CSV con separador `;` (formato propio de UCI).
- Separación de features (X) y target (`quality`).
- División train/test con `test_size=0.2` y `random_state=42` para el entrenamiento.
- División train/test con `random_state=99` distinto para la validación, simulando datos externos no vistos por el modelo.

---

## Estructura del proyecto

```
mlflow-deploy/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml          # Workflow de GitHub Actions
├── data/
│   └── winequality-red.csv        # Dataset externo (UCI)
├── mlruns/                        # Tracking local de MLflow (generado)
├── train.py                       # Script de entrenamiento + registro
├── validate.py                    # Script de validación contra umbrales
├── requirements.txt               # Dependencias del proyecto
├── Makefile                       # Comandos automatizados
├── .gitignore
└── README.md
```

---

## Requisitos

- Python 3.10 o superior
- pip
- Git

Las dependencias específicas del proyecto se listan en `requirements.txt`:

```
mlflow>=2.10.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

---

## Instalación y uso local

### 1. Clonar el repositorio

```bash
git clone https://github.com/cqdirecly/mlflow-deploy.git
cd mlflow-deploy
```

### 2. Crear y activar un entorno virtual

**Linux / macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
make install
```

O directamente:
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el pipeline

**Pipeline completo (train + validate):**
```bash
make pipeline
```

**Solo entrenamiento:**
```bash
make train
```

**Solo validación (requiere haber entrenado antes):**
```bash
make validate
```

### 5. Visualizar resultados en MLflow UI

```bash
make mlflow-ui
```

Luego abrir en el navegador: `http://127.0.0.1:5000`

### 6. Limpiar artefactos generados

```bash
make clean
```

---

## Pipeline de CI/CD

### Disparadores

El workflow `mlflow-ci.yml` se ejecuta automáticamente en los siguientes eventos:

- **Push** a la rama `main`
- **Pull Request** dirigido a la rama `main`
- **Ejecución manual** (workflow_dispatch) desde la pestaña Actions

### Etapas del workflow

| # | Etapa | Descripción |
|---|-------|-------------|
| 1 | Clonar repositorio | Descarga el código fuente en el runner |
| 2 | Configurar Python | Instala Python 3.10 con caché de pip |
| 3 | Instalar dependencias | Ejecuta `make install` |
| 4 | Entrenar modelo | Ejecuta `make train` (genera `mlruns/`, `model.pkl`) |
| 5 | Validar modelo | Ejecuta `make validate` (verifica umbrales) |
| 6 | Subir modelo | Publica `model.pkl` y `latest_run_id.txt` como artefacto |
| 7 | Subir tracking | Publica la carpeta `mlruns/` completa como artefacto |

### Resultado esperado

Si todas las etapas pasan exitosamente, el workflow queda en estado **success** (verde) y se generan dos artefactos descargables desde la pestaña Actions de GitHub:

- **`modelo-validado`**: contiene el modelo serializado y la referencia del run.
- **`mlruns-tracking`**: contiene el tracking completo de MLflow para auditoría.

---

## Detalle de componentes

### `train.py`

Script modular que ejecuta el pipeline de entrenamiento. Sus responsabilidades son:

- Cargar el dataset Wine Quality desde el CSV local.
- Dividir los datos en train/test.
- Entrenar un modelo `RandomForestRegressor` con hiperparámetros definidos.
- Calcular métricas de evaluación (MSE, RMSE, MAE, R²).
- Configurar el tracking URI de MLflow y crear/recuperar el experimento.
- Registrar en MLflow:
  - Parámetros del modelo (`n_estimators`, `max_depth`, `random_state`, etc.).
  - Métricas calculadas.
  - El modelo entrenado con **signature** e **input_example**.
- Guardar el modelo como `model.pkl` (artefacto adicional).
- Persistir el `run_id` en `latest_run_id.txt` para uso posterior por `validate.py`.

El código está organizado en funciones documentadas: `load_data`, `prepare_data`, `train_model`, `evaluate_model`, `setup_mlflow_tracking`, `get_or_create_experiment`, `log_to_mlflow` y `main`.

### `validate.py`

Script modular que ejecuta la validación del modelo registrado. Sus responsabilidades son:

- Configurar el tracking URI de MLflow.
- Leer el `run_id` desde `latest_run_id.txt`.
- Cargar el modelo desde MLflow usando `mlflow.sklearn.load_model(f"runs:/{run_id}/model")`.
- Cargar datos externos de validación con un `random_state` diferente al de entrenamiento.
- Evaluar el modelo y calcular métricas.
- Validar contra umbrales (`MSE_THRESHOLD = 1.0`, `R2_THRESHOLD = 0.3`).
- Salir con código `0` (éxito) o `1` (fallo) para integración con CI/CD.

Las funciones están organizadas como: `get_run_id`, `setup_mlflow_tracking`, `load_model_from_mlflow`, `load_validation_data`, `evaluate_model`, `validate_metrics` y `main`.

### `Makefile`

Define los comandos de automatización del proyecto. Cada target está documentado con comentarios:

| Target | Descripción |
|--------|-------------|
| `make help` | Muestra ayuda con la lista de comandos disponibles |
| `make install` | Instala las dependencias desde `requirements.txt` |
| `make train` | Ejecuta `train.py` para entrenar y registrar el modelo |
| `make validate` | Ejecuta `validate.py` para validar el modelo |
| `make pipeline` | Ejecuta `train` y `validate` en secuencia |
| `make mlflow-ui` | Lanza la UI de MLflow en el navegador |
| `make clean` | Elimina artefactos generados localmente |

### `mlflow-ci.yml`

Workflow de GitHub Actions que orquesta el pipeline completo en cada push a `main`. Utiliza acciones oficiales de GitHub (`actions/checkout@v4`, `actions/setup-python@v5`, `actions/upload-artifact@v4`) y ejecuta los targets del Makefile.

---

## Métricas y umbrales de validación

### Métricas calculadas

El proyecto registra cuatro métricas estándar para evaluación de modelos de regresión:

- **MSE** (Mean Squared Error): error cuadrático medio.
- **RMSE** (Root Mean Squared Error): raíz del error cuadrático medio (en unidades del target).
- **MAE** (Mean Absolute Error): error absoluto medio.
- **R²** (Coeficiente de determinación): proporción de varianza explicada por el modelo.

### Umbrales de validación

`validate.py` aplica los siguientes criterios de calidad:

| Métrica | Umbral | Criterio |
|---------|--------|----------|
| MSE | ≤ 1.0 | El error cuadrático medio debe ser bajo |
| R² | ≥ 0.3 | El modelo debe explicar al menos 30% de la varianza |

Si alguno de los criterios no se cumple, `validate.py` retorna código de salida `1` y el workflow de CI/CD falla, evitando la publicación de un modelo deficiente.

---

## Trazabilidad y auditoría

El proyecto garantiza la trazabilidad y reproducibilidad del modelo a través de varios mecanismos:

### Tracking de MLflow

Cada ejecución de `train.py` genera un run en MLflow que registra:

- **Parámetros:** todos los hiperparámetros y metadatos del experimento.
- **Métricas:** MSE, RMSE, MAE y R² del set de prueba.
- **Modelo:** serializado en formato MLflow con `mlflow.sklearn.log_model`.
- **Signature:** firma generada con `infer_signature` que documenta los tipos y nombres de las features de entrada y la salida.
- **Input example:** muestra de cinco filas reales de entrada para facilitar pruebas y documentación.

### Artefactos de GitHub Actions

Tras cada ejecución exitosa del workflow, GitHub Actions publica dos artefactos descargables:

- **`modelo-validado`:** contiene el modelo entrenado en formato `.pkl` y la referencia al `run_id`.
- **`mlruns-tracking`:** contiene la carpeta completa `mlruns/` con todos los runs registrados, lista para inspección.

Estos artefactos quedan disponibles durante 30 días en la pestaña Actions del repositorio y permiten reproducir cualquier ejecución del pipeline.

---

## Autor

**Christian Quimbay**
Maestría en Ciencia de Datos — Curso de MLOps

---

## Licencia

Proyecto académico desarrollado con fines educativos.