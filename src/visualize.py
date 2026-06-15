import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    CV_RESULTS_FILE,
    MODEL_FILE,
    METRICS_FILE,
    OUTPUTS_DIR,
    PROCESSED_DATA_FILE,
)


def plot_metrics():
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    metric_values = [metrics[m] for m in metric_names]

    plt.figure(figsize=(8, 5))
    plt.bar(metric_names, metric_values)
    plt.ylim(0, 1)
    plt.title("Métricas del modelo en test")
    plt.xlabel("Métrica")
    plt.ylabel("Valor")

    output_path = OUTPUTS_DIR / "metricas_modelo.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Gráfico guardado en: {output_path}")


def plot_confusion_matrix():
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = metrics["confusion_matrix"]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de confusión")
    plt.colorbar()

    plt.xticks([0, 1], ["Pred: No fatal", "Pred: Fatal"])
    plt.yticks([0, 1], ["Real: No fatal", "Real: Fatal"])

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.xlabel("Predicción")
    plt.ylabel("Valor real")

    output_path = OUTPUTS_DIR / "matriz_confusion.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Gráfico guardado en: {output_path}")


def plot_class_balance():
    df = pd.read_csv(PROCESSED_DATA_FILE, low_memory=False)

    class_counts = df["fatal"].value_counts().sort_index()

    labels = ["No fatal (0)", "Fatal (1)"]
    values = [class_counts.get(0, 0), class_counts.get(1, 0)]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title("Distribución de clases")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de casos")

    output_path = OUTPUTS_DIR / "desbalance_clases.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Gráfico guardado en: {output_path}")


def plot_cross_validation_results():
    if not CV_RESULTS_FILE.exists():
        print("Aviso: no se encontró cv_results.csv. Se omite este gráfico.")
        return

    cv_results = pd.read_csv(CV_RESULTS_FILE)

    if "mean_test_f1" not in cv_results.columns:
        print("Aviso: cv_results.csv no contiene mean_test_f1. Se omite este gráfico.")
        return

    top_results = cv_results.sort_values("rank_test_f1").head(8).copy()

    labels = []
    for _, row in top_results.iterrows():
        classifier = row.get("param_classifier", "modelo")
        if "RandomForestClassifier" in classifier:
            label = "Random Forest"
            if "param_classifier__max_depth" in row:
                label += f" | depth={row['param_classifier__max_depth']}"
            if "param_classifier__n_estimators" in row:
                label += f" | n={row['param_classifier__n_estimators']}"
        elif "LogisticRegression" in classifier:
            label = "Logistic Regression"
            if "param_classifier__C" in row:
                label += f" | C={row['param_classifier__C']}"
        else:
            label = str(classifier)[:30]
        labels.append(label)

    plt.figure(figsize=(10, 6))
    plt.barh(labels, top_results["mean_test_f1"])
    plt.xlim(0, 1)
    plt.title("Mejores resultados de validación cruzada")
    plt.xlabel("F1-score promedio en CV")
    plt.ylabel("Modelo e hiperparámetros")
    plt.gca().invert_yaxis()

    output_path = OUTPUTS_DIR / "validacion_cruzada_f1.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Gráfico guardado en: {output_path}")


def plot_feature_importance():
    if not MODEL_FILE.exists():
        print("Aviso: no se encontró el modelo entrenado. Se omite importancia de variables.")
        return

    model = joblib.load(MODEL_FILE)
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        print(
            "Aviso: el mejor modelo no expone feature_importances_. "
            "Se omite este gráfico."
        )
        return

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("Importancia de variables - Top 10")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.gca().invert_yaxis()

    output_path = OUTPUTS_DIR / "importancia_variables.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Gráfico guardado en: {output_path}")


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_metrics()
    plot_confusion_matrix()
    plot_class_balance()
    plot_cross_validation_results()
    plot_feature_importance()
    print("Visualizaciones generadas correctamente.")


if __name__ == "__main__":
    main()
