import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BASE_DIR,
    CATEGORICAL_FEATURES,
    CV_RESULTS_FILE,
    CV_SPLITS,
    MODEL_FILE,
    NUMERIC_FEATURES,
    PROCESSED_DATA_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_FEATURES_FILE,
    TEST_SIZE,
    TEST_TARGET_FILE,
)
from src.db import insert_model_config


def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_FILE, low_memory=False)

def build_preprocessor():
    """Construye el preprocesador para variables numéricas y categóricas."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline():
    """Define el pipeline base: preprocesamiento + clasificador."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def build_param_grid():
    """Define los modelos e hiperparámetros que se comparan con GridSearchCV."""
    return [
        {
            "classifier": [
                LogisticRegression(max_iter=1000, class_weight="balanced")
            ],
            "classifier__C": [0.1, 1, 10],
        },
        {
            "classifier": [
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                )
            ],
            "classifier__n_estimators": [100],
            "classifier__max_depth": [None, 10],
        },
    ]


def validate_columns(df):
    expected_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Faltan columnas en el dataset procesado: {missing_columns}"
        )


def save_cv_results(grid_search):
    """Guarda un resumen de los resultados de validación cruzada."""
    cv_results = pd.DataFrame(grid_search.cv_results_)

    columns_to_keep = [
        "param_classifier",
        "param_classifier__C",
        "param_classifier__n_estimators",
        "param_classifier__max_depth",
        "mean_test_accuracy",
        "mean_test_precision",
        "mean_test_recall",
        "mean_test_f1",
        "std_test_f1",
        "rank_test_f1",
    ]

    existing_columns = [col for col in columns_to_keep if col in cv_results.columns]
    cv_summary = cv_results[existing_columns].copy()

    for col in cv_summary.columns:
        if col.startswith("param_"):
            cv_summary[col] = cv_summary[col].astype(str)

    CV_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cv_summary.sort_values("rank_test_f1").to_csv(CV_RESULTS_FILE, index=False)

    return cv_summary


def serialize_best_params(best_params):
    """Convierte los mejores hiperparámetros a un formato apto para JSON/MongoDB."""
    best_params_serializable = {}

    for key, value in best_params.items():
        if key == "classifier":
            best_params_serializable[key] = type(value).__name__
        else:
            best_params_serializable[key] = value

    return best_params_serializable


def train_models():
    df = load_processed_data()

    print("Columnas del dataset procesado:")
    print(df.columns.tolist())

    validate_columns(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv_strategy = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0)
    }

    grid_search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=build_param_grid(),
        cv=cv_strategy,
        scoring=scoring,
        refit="f1",
        n_jobs=1,
        return_train_score=True,
        error_score="raise",
    )

    grid_search.fit(X_train, y_train)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid_search.best_estimator_, MODEL_FILE)

    TEST_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    X_test.to_csv(TEST_FEATURES_FILE, index=False)
    y_test.to_csv(TEST_TARGET_FILE, index=False)

    cv_summary = save_cv_results(grid_search)
    best_params_serializable = serialize_best_params(grid_search.best_params_)

    config_doc = {
        "target_column": TARGET_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_model": type(
            grid_search.best_estimator_.named_steps["classifier"]
        ).__name__,
        "best_params": best_params_serializable,
        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": CV_SPLITS,
            "shuffle": True,
            "random_state": RANDOM_STATE,
            "scoring": ["accuracy", "precision", "recall", "f1"],
            "refit_metric": "f1",
        },
        "cv_best_f1_score": float(grid_search.best_score_),
        "cv_results_file": str(CV_RESULTS_FILE.relative_to(BASE_DIR)),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "stratify": True,
    }

    insert_model_config(config_doc)

    return (
        grid_search.best_estimator_,
        X_test,
        y_test,
        best_params_serializable,
        grid_search.best_score_,
        cv_summary,
    )


def main():
    best_model, X_test, y_test, best_params, best_score, cv_summary = train_models()

    print("\nEntrenamiento finalizado.")
    print("Mejor modelo:", type(best_model.named_steps["classifier"]).__name__)
    print("Mejores parámetros:", best_params)
    print("Mejor F1 promedio en CV:", best_score)
    print(f"Modelo guardado en: {MODEL_FILE}")
    print(f"Resultados de validación cruzada guardados en: {CV_RESULTS_FILE}")
    print(f"X_test guardado en: {TEST_FEATURES_FILE}")
    print(f"y_test guardado en: {TEST_TARGET_FILE}")


if __name__ == "__main__":
    main()
