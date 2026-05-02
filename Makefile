# ==============================================================================
# Makefile para Pipeline CI/CD con MLflow
# ==============================================================================
# Proyecto: Wine Quality Regression - Pipeline CI/CD con MLflow
# Autor:    Christian Quimbay
# Curso:    MLOps - Maestria en Ciencia de Datos
#
# Uso:
#   make help       - Muestra esta ayuda con la lista de comandos disponibles
#   make install    - Instala las dependencias del proyecto
#   make train      - Entrena el modelo y lo registra en MLflow
#   make validate   - Valida el modelo registrado contra umbrales de calidad
#   make pipeline   - Ejecuta el pipeline completo: train + validate
#   make mlflow-ui  - Lanza la UI de MLflow en http://127.0.0.1:5000
#   make clean      - Elimina artefactos generados (mlruns, model.pkl, etc.)
# ==============================================================================

# Variable para apuntar al ejecutable de Python (compatible con venv local)
PYTHON := python

.PHONY: help install train validate pipeline mlflow-ui clean

# ------------------------------------------------------------------------------
# help: Muestra la ayuda con todos los comandos disponibles
# ------------------------------------------------------------------------------
help:
	@echo "==================================================================="
	@echo "  Pipeline CI/CD con MLflow - Wine Quality Regression"
	@echo "==================================================================="
	@echo "  make install    - Instala las dependencias del proyecto"
	@echo "  make train      - Entrena el modelo y lo registra en MLflow"
	@echo "  make validate   - Valida el modelo registrado"
	@echo "  make pipeline   - Ejecuta train + validate (flujo completo)"
	@echo "  make mlflow-ui  - Lanza la UI de MLflow"
	@echo "  make clean      - Elimina artefactos generados"
	@echo "==================================================================="

# ------------------------------------------------------------------------------
# install: Instala las dependencias del proyecto desde requirements.txt
# ------------------------------------------------------------------------------
# Actualiza pip y luego instala todas las librerias necesarias
# (mlflow, scikit-learn, pandas, numpy, joblib).
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# ------------------------------------------------------------------------------
# train: Entrena el modelo y lo registra en MLflow
# ------------------------------------------------------------------------------
# Ejecuta train.py que realiza:
#   1. Carga el dataset Wine Quality (Red Wine) desde data/winequality-red.csv.
#   2. Divide los datos en train/test.
#   3. Entrena un modelo RandomForestRegressor.
#   4. Calcula metricas (MSE, RMSE, MAE, R2).
#   5. Registra el experimento, parametros, metricas y modelo en MLflow,
#      incluyendo signature e input_example para trazabilidad.
#   6. Guarda el run_id en latest_run_id.txt para que validate.py lo use.
train:
	$(PYTHON) train.py

# ------------------------------------------------------------------------------
# validate: Valida el modelo registrado contra umbrales de calidad
# ------------------------------------------------------------------------------
# Ejecuta validate.py que realiza:
#   1. Lee el run_id del entrenamiento mas reciente.
#   2. Carga el modelo registrado en MLflow desde el run.
#   3. Carga datos externos de validacion con un random_state distinto.
#   4. Evalua el modelo con MSE, RMSE, MAE, R2.
#   5. Valida que las metricas cumplan los umbrales (MSE <= 1.0, R2 >= 0.3).
#   6. Sale con codigo 0 si pasa, 1 si falla (para integracion CI/CD).
validate:
	$(PYTHON) validate.py

# ------------------------------------------------------------------------------
# pipeline: Ejecuta el pipeline completo (train + validate)
# ------------------------------------------------------------------------------
# Util para correr todo el flujo localmente con un solo comando.
# Si el entrenamiento falla, validate no se ejecuta.
pipeline: train validate

# ------------------------------------------------------------------------------
# mlflow-ui: Lanza la UI de MLflow en el navegador
# ------------------------------------------------------------------------------
# Abre la UI de MLflow apuntando al directorio mlruns/ local.
# Acceder en: http://127.0.0.1:5000
# Detener con Ctrl+C.
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns

# ------------------------------------------------------------------------------
# clean: Elimina artefactos generados por el pipeline
# ------------------------------------------------------------------------------
# Util para empezar desde cero. Borra:
#   - Carpeta mlruns/ (experimentos y artefactos de MLflow).
#   - model.pkl (modelo serializado).
#   - latest_run_id.txt (referencia al ultimo run).
#   - __pycache__/ (cache de Python).
clean:
	@echo "Limpiando artefactos generados..."
	@if exist mlruns rmdir /s /q mlruns
	@if exist model.pkl del /q model.pkl
	@if exist latest_run_id.txt del /q latest_run_id.txt
	@if exist __pycache__ rmdir /s /q __pycache__
	@echo "Limpieza completada."