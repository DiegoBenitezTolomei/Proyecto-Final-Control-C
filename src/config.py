from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"

RAW_DATA_FILE = DATA_RAW_DIR / "siniestros_viales_victimas.xlsx"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "dataset_limpio.csv"
MODEL_FILE = MODELS_DIR / "best_model.pkl"
METRICS_FILE = METRICS_DIR / "metricas.json"
CV_RESULTS_FILE = METRICS_DIR / "cv_results.csv"
PREDICTIONS_FILE = OUTPUTS_DIR / "predicciones_test.csv"

TEST_FEATURES_FILE = OUTPUTS_DIR / "X_test.csv"
TEST_TARGET_FILE = OUTPUTS_DIR / "y_test.csv"

MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "accidentes_viales"

COLL_INPUT = "datos_entrada"
COLL_RESULTS = "resultados_modelo"
COLL_CONFIG = "configuracion_modelo"

TARGET_COLUMN = "fatal"

NUMERIC_FEATURES = ["anio", "edad_victima", "mes", "dia_semana"]
CATEGORICAL_FEATURES = [
    "sexo_victima",
    "modo_desplazamiento_victima",
    "rol_victima", 
    "clima"
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_SPLITS = 5

THRESHOLD_FATAL = 0.2