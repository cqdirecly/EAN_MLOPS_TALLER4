"""
train.py
Pipeline de entrenamiento y registro de modelo con MLflow.

Este script:
1. Carga el dataset Wine Quality (Red Wine) desde un archivo CSV local.
2. Realiza un split train/test.
3. Entrena un modelo RandomForestRegressor.
4. Evalua el modelo con multiples metricas (MSE, RMSE, MAE, R2).
5. Registra el experimento, parametros, metricas y modelo en MLflow,
   incluyendo signature e input_example para garantizar trazabilidad
   y reproducibilidad del modelo.

Autor: Christian Quimbay
Curso: MLOps - Maestria en Ciencia de Datos
"""

import logging
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------- Configuracion ----------------------------

# Configuracion de logging para mensajes claros y trazables
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constantes del proyecto
DATA_PATH = Path("data") / "winequality-red.csv"
TARGET_COL = "quality"
RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "wine-quality-regression"
MODEL_ARTIFACT_PATH = "model"
MODEL_PKL_PATH = "model.pkl"
RUN_ID_FILE = "latest_run_id.txt"

# Hiperparametros del modelo
N_ESTIMATORS = 100
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 2


# ---------------------------- Funciones ----------------------------

def load_data(data_path: Path) -> pd.DataFrame:
    """Carga el dataset Wine Quality desde un archivo CSV.

    Args:
        data_path: Ruta al archivo CSV.

    Returns:
        DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {data_path}. "
            "Asegurate de haber descargado winequality-red.csv en la carpeta data/."
        )

    logger.info(f"Cargando dataset desde {data_path}")
    df = pd.read_csv(data_path, sep=";")
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_state: int,
) -> tuple:
    """Divide el dataset en features/target y train/test.

    Args:
        df: DataFrame original.
        target_col: Nombre de la columna objetivo.
        test_size: Proporcion del test set.
        random_state: Semilla para reproducibilidad.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(
        f"Train: {X_train.shape[0]} muestras, Test: {X_test.shape[0]} muestras"
    )
    return X_train, X_test, y_train, y_test


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    random_state: int,
) -> RandomForestRegressor:
    """Entrena un modelo RandomForestRegressor.

    Args:
        X_train: Features de entrenamiento.
        y_train: Target de entrenamiento.
        n_estimators: Numero de arboles del bosque.
        max_depth: Profundidad maxima de cada arbol.
        min_samples_split: Minimo de muestras para dividir un nodo.
        random_state: Semilla para reproducibilidad.

    Returns:
        Modelo entrenado.
    """
    logger.info("Iniciando entrenamiento del modelo RandomForestRegressor")
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    logger.info("Entrenamiento completado")
    return model


def evaluate_model(
    model: RandomForestRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evalua el modelo con multiples metricas de regresion.

    Args:
        model: Modelo entrenado.
        X_test: Features de prueba.
        y_test: Target de prueba.

    Returns:
        Diccionario con las metricas (mse, rmse, mae, r2).
    """
    predictions = model.predict(X_test)

    metrics = {
        "mse": float(mean_squared_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }

    logger.info("Metricas del modelo en test:")
    for name, value in metrics.items():
        logger.info(f"  {name.upper()}: {value:.4f}")

    return metrics


def setup_mlflow_tracking() -> None:
    """Configura el tracking URI de MLflow apuntando a mlruns/ local.

    Esto asegura que tanto en ejecucion local como en CI/CD el tracking
    se guarde dentro del workspace del proyecto.
    """
    workspace_dir = Path.cwd()
    mlruns_dir = workspace_dir / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)

    tracking_uri = mlruns_dir.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI configurado en: {tracking_uri}")


def get_or_create_experiment(experiment_name: str) -> str:
    """Obtiene o crea un experimento en MLflow.

    Args:
        experiment_name: Nombre del experimento.

    Returns:
        ID del experimento.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(
            f"Experimento '{experiment_name}' creado con ID: {experiment_id}"
        )
    else:
        experiment_id = experiment.experiment_id
        logger.info(
            f"Usando experimento existente '{experiment_name}' con ID: {experiment_id}"
        )
    return experiment_id


def log_to_mlflow(
    model: RandomForestRegressor,
    metrics: dict,
    params: dict,
    X_train: pd.DataFrame,
    experiment_id: str,
) -> str:
    """Registra el modelo, parametros y metricas en MLflow.

    El modelo se registra con signature e input_example para garantizar
    trazabilidad y permitir validacion automatica de inputs en futuras
    inferencias.

    Args:
        model: Modelo entrenado.
        metrics: Diccionario de metricas.
        params: Diccionario de hiperparametros y metadatos.
        X_train: Features de entrenamiento (para signature e input_example).
        experiment_id: ID del experimento de MLflow.

    Returns:
        run_id del run de MLflow recien creado.
    """
    with mlflow.start_run(experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        logger.info(f"Iniciando run de MLflow con ID: {run_id}")

        # Registrar parametros
        mlflow.log_params(params)

        # Registrar metricas
        mlflow.log_metrics(metrics)

        # Generar signature e input_example (criterio clave de nota destacable)
        predictions = model.predict(X_train)
        signature = infer_signature(X_train, predictions)
        input_example = X_train.head(5)

        # Registrar el modelo con signature e input_example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=MODEL_ARTIFACT_PATH,
            signature=signature,
            input_example=input_example,
        )
        logger.info(
            f"Modelo registrado en MLflow con artifact_path='{MODEL_ARTIFACT_PATH}'"
        )

        # Guardar el modelo tambien como .pkl (artefacto adicional para CI/CD)
        joblib.dump(model, MODEL_PKL_PATH)
        logger.info(f"Modelo guardado adicionalmente como '{MODEL_PKL_PATH}'")

        # Guardar el run_id para que validate.py pueda usarlo
        with open(RUN_ID_FILE, "w", encoding="utf-8") as f:
            f.write(run_id)
        logger.info(f"run_id guardado en '{RUN_ID_FILE}'")

        return run_id


# ---------------------------- Main ----------------------------

def main() -> None:
    """Pipeline principal de entrenamiento y registro."""
    try:
        # 1. Cargar datos
        df = load_data(DATA_PATH)

        # 2. Preparar datos
        X_train, X_test, y_train, y_test = prepare_data(
            df,
            target_col=TARGET_COL,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

        # 3. Entrenar modelo
        model = train_model(
            X_train,
            y_train,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            random_state=RANDOM_STATE,
        )

        # 4. Evaluar modelo
        metrics = evaluate_model(model, X_test, y_test)

        # 5. Configurar MLflow tracking
        setup_mlflow_tracking()
        experiment_id = get_or_create_experiment(EXPERIMENT_NAME)

        # 6. Registrar en MLflow
        params = {
            "model_type": "RandomForestRegressor",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_split": MIN_SAMPLES_SPLIT,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "dataset": "Wine Quality Red - UCI Repository",
        }
        run_id = log_to_mlflow(
            model=model,
            metrics=metrics,
            params=params,
            X_train=X_train,
            experiment_id=experiment_id,
        )

        logger.info(
            f"Pipeline de entrenamiento completado exitosamente. Run ID: {run_id}"
        )
        print(
            f"Modelo registrado correctamente. "
            f"MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}"
        )

    except FileNotFoundError as e:
        logger.error(f"Error de archivo: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error inesperado durante el entrenamiento: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()