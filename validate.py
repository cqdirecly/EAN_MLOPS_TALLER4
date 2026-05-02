"""
validate.py
Pipeline de validacion del modelo registrado en MLflow.

Este script:
1. Lee el run_id del entrenamiento mas reciente desde 'latest_run_id.txt'.
2. Carga el modelo registrado en MLflow desde el run correspondiente.
3. Carga datos externos de validacion (Wine Quality desde CSV) y los procesa
   con un random_state distinto al de entrenamiento, simulando datos nuevos.
4. Evalua el modelo con multiples metricas (MSE, RMSE, MAE, R2).
5. Valida si las metricas cumplen los umbrales de calidad definidos.
6. Sale con codigo 0 (exito) o 1 (fallo) para integracion con CI/CD.

Autor: Christian Quimbay
Curso: MLOps - Maestria en Ciencia de Datos
"""

import logging
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------- Configuracion ----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constantes del proyecto
DATA_PATH = Path("data") / "winequality-red.csv"
TARGET_COL = "quality"
MODEL_ARTIFACT_PATH = "model"
RUN_ID_FILE = "latest_run_id.txt"

# Random state diferente al de entrenamiento para simular datos externos
VALIDATION_RANDOM_STATE = 99
VALIDATION_TEST_SIZE = 0.3

# Umbrales de validacion (criterios de calidad del modelo)
MSE_THRESHOLD = 1.0   # MSE maximo aceptable
R2_THRESHOLD = 0.3    # R2 minimo aceptable


# ---------------------------- Funciones ----------------------------

def get_run_id(run_id_file: str) -> str:
    """Lee el run_id del entrenamiento mas reciente.

    Args:
        run_id_file: Ruta al archivo que contiene el run_id.

    Returns:
        run_id como string.

    Raises:
        FileNotFoundError: Si el archivo no existe.
    """
    run_id_path = Path(run_id_file)
    if not run_id_path.exists():
        raise FileNotFoundError(
            f"No se encontro {run_id_file}. "
            "Asegurate de haber ejecutado 'make train' antes de validar."
        )

    run_id = run_id_path.read_text(encoding="utf-8").strip()
    logger.info(f"run_id leido: {run_id}")
    return run_id


def setup_mlflow_tracking() -> None:
    """Configura el tracking URI de MLflow apuntando a mlruns/ local.

    Debe coincidir con la configuracion usada en train.py para poder
    acceder al modelo registrado.
    """
    workspace_dir = Path.cwd()
    mlruns_dir = workspace_dir / "mlruns"

    if not mlruns_dir.exists():
        raise FileNotFoundError(
            f"No se encontro el directorio mlruns/ en {workspace_dir}. "
            "Asegurate de haber ejecutado 'make train' antes de validar."
        )

    tracking_uri = mlruns_dir.resolve().as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI configurado en: {tracking_uri}")


def load_model_from_mlflow(run_id: str, artifact_path: str):
    """Carga el modelo registrado en MLflow desde un run especifico.

    Args:
        run_id: ID del run de MLflow.
        artifact_path: Path del artefacto del modelo dentro del run.

    Returns:
        Modelo de scikit-learn cargado y listo para predecir.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info(f"Cargando modelo desde MLflow URI: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Modelo cargado exitosamente desde MLflow")
    return model


def load_validation_data(
    data_path: Path,
    target_col: str,
    test_size: float,
    random_state: int,
) -> tuple:
    """Carga datos externos de validacion desde el CSV.

    Usa un random_state distinto al de entrenamiento para simular
    datos nuevos no vistos por el modelo durante el entrenamiento.

    Args:
        data_path: Ruta al archivo CSV.
        target_col: Nombre de la columna objetivo.
        test_size: Proporcion del set de validacion.
        random_state: Semilla para el split.

    Returns:
        Tupla (X_val, y_val) con los datos de validacion.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {data_path}."
        )

    logger.info(f"Cargando datos de validacion desde {data_path}")
    df = pd.read_csv(data_path, sep=";")
    logger.info(f"Dataset completo: {df.shape[0]} filas, {df.shape[1]} columnas")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split con random_state distinto al de entrenamiento
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Set de validacion: {X_val.shape[0]} muestras")

    return X_val, y_val


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Evalua el modelo con multiples metricas de regresion.

    Args:
        model: Modelo cargado desde MLflow.
        X_val: Features de validacion.
        y_val: Target de validacion.

    Returns:
        Diccionario con las metricas (mse, rmse, mae, r2).
    """
    predictions = model.predict(X_val)

    metrics = {
        "mse": float(mean_squared_error(y_val, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_val, predictions))),
        "mae": float(mean_absolute_error(y_val, predictions)),
        "r2": float(r2_score(y_val, predictions)),
    }

    logger.info("Metricas del modelo en datos de validacion:")
    for name, value in metrics.items():
        logger.info(f"  {name.upper()}: {value:.4f}")

    return metrics


def validate_metrics(
    metrics: dict,
    mse_threshold: float,
    r2_threshold: float,
) -> bool:
    """Valida si las metricas cumplen los umbrales de calidad definidos.

    Args:
        metrics: Diccionario con las metricas calculadas.
        mse_threshold: Umbral maximo aceptable de MSE.
        r2_threshold: Umbral minimo aceptable de R2.

    Returns:
        True si el modelo cumple ambos criterios, False en caso contrario.
    """
    logger.info("Validando criterios de calidad del modelo:")
    logger.info(f"  Umbral MSE maximo: {mse_threshold}")
    logger.info(f"  Umbral R2 minimo: {r2_threshold}")

    mse_ok = metrics["mse"] <= mse_threshold
    r2_ok = metrics["r2"] >= r2_threshold

    logger.info(
        f"  MSE = {metrics['mse']:.4f} {'OK' if mse_ok else 'FALLA'}"
    )
    logger.info(
        f"  R2 = {metrics['r2']:.4f} {'OK' if r2_ok else 'FALLA'}"
    )

    return mse_ok and r2_ok


# ---------------------------- Main ----------------------------

def main() -> None:
    """Pipeline principal de validacion."""
    try:
        # 1. Configurar MLflow tracking
        setup_mlflow_tracking()

        # 2. Leer run_id del entrenamiento mas reciente
        run_id = get_run_id(RUN_ID_FILE)

        # 3. Cargar modelo registrado en MLflow
        model = load_model_from_mlflow(run_id, MODEL_ARTIFACT_PATH)

        # 4. Cargar datos externos de validacion
        X_val, y_val = load_validation_data(
            data_path=DATA_PATH,
            target_col=TARGET_COL,
            test_size=VALIDATION_TEST_SIZE,
            random_state=VALIDATION_RANDOM_STATE,
        )

        # 5. Evaluar modelo
        metrics = evaluate_model(model, X_val, y_val)

        # 6. Validar contra umbrales
        passed = validate_metrics(
            metrics=metrics,
            mse_threshold=MSE_THRESHOLD,
            r2_threshold=R2_THRESHOLD,
        )

        # 7. Resultado final
        if passed:
            print(
                f"El modelo cumple los criterios de calidad. "
                f"MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}"
            )
            sys.exit(0)
        else:
            print(
                f"El modelo NO cumple los criterios de calidad. "
                f"MSE: {metrics['mse']:.4f}, R2: {metrics['r2']:.4f}"
            )
            sys.exit(1)

    except FileNotFoundError as e:
        logger.error(f"Error de archivo: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error inesperado durante la validacion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()