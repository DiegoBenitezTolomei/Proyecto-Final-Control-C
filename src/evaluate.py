import json
from xml.parsers.expat import model

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    BASE_DIR,
    METRICS_FILE,
    MODEL_FILE,
    PREDICTIONS_FILE,
    TEST_FEATURES_FILE,
    TEST_TARGET_FILE,
    THRESHOLD_FATAL,
)
from src.db import insert_model_results


def get_positive_class_probability(model, X_test):
    """Obtiene la probabilidad de fatalidad si el modelo soporta predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]

    return [None] * len(X_test)


def save_predictions(X_test, y_test, y_pred, y_proba):
    """Guarda predicciones finales y probabilidad estimada de fatalidad."""
    predictions = X_test.copy().reset_index(drop=True)
    predictions["fatal_real"] = pd.Series(y_test).reset_index(drop=True)
    predictions["fatal_predicho"] = y_pred
    predictions["probabilidad_fatal"] = y_proba
    predictions["umbral_decision"] = THRESHOLD_FATAL

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    return predictions


def evaluate_model():
     # Cargar modelo entrenado
    model = joblib.load(MODEL_FILE)

    # Cargar conjunto de prueba guardado desde train.py
    X_test = pd.read_csv(TEST_FEATURES_FILE)
    y_test = pd.read_csv(TEST_TARGET_FILE).squeeze()

    # Probabilidad de clase fatal
    y_proba = model.predict_proba(X_test)[:, 1]

    # Aplicar umbral de decisión elegido
    y_pred = (y_proba >= THRESHOLD_FATAL).astype(int)


    predictions = save_predictions(X_test, y_test, y_pred, y_proba)

    metrics = {
        "threshold_fatal": THRESHOLD_FATAL,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "predictions_file": str(PREDICTIONS_FILE.relative_to(BASE_DIR)),
    }

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    result_doc = {
        "modelo": type(model.named_steps["classifier"]).__name__,
        "evaluado_sobre": "test",
        "metricas": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "predictions_file": str(PREDICTIONS_FILE.relative_to(BASE_DIR)),
        "cantidad_predicciones": int(len(predictions)),
    }

    insert_model_results(result_doc)

    return metrics


def main():
    metrics = evaluate_model()
    print("Evaluación finalizada sobre conjunto de prueba.")
    print(f"Predicciones guardadas en: {PREDICTIONS_FILE}")
    print(json.dumps(metrics, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
