import json

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

from src.config import (
    CV_RESULTS_FILE,
    MODEL_FILE,
    METRICS_FILE,
    OUTPUTS_DIR,
    PROCESSED_DATA_FILE,
)

# ── Paleta de colores consistente en todos los gráficos ─────────────────────
COLOR_PRIMARY = "#2C7BB6"
COLOR_FATAL = "#D7191C"
COLOR_NO_FATAL = "#1A9641"
CMAP_HEATMAP = "Blues"


def plot_metrics():
    """Gráfico de barras con las métricas de clasificación en el conjunto test."""
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    metric_values = [metrics.get(m) for m in metric_names]

    # Filtrar métricas que pueden ser None (ej. AUC cuando no aplica)
    pairs = [(l, v) for l, v in zip(metric_labels, metric_values) if v is not None]
    labels, values = zip(*pairs)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=COLOR_PRIMARY, edgecolor="white", width=0.6)
    ax.set_ylim(0, 1.1)
    ax.set_title("Métricas del modelo en conjunto de test", fontsize=14, pad=12)
    ax.set_xlabel("Métrica")
    ax.set_ylabel("Valor")
    ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--", label="Umbral 0.5")
    ax.legend(fontsize=9)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    output_path = OUTPUTS_DIR / "metricas_modelo.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


def plot_confusion_matrix():
    """Heatmap de la matriz de confusión con etiquetas absolutas y porcentajes."""
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = np.array(metrics["confusion_matrix"])
    total = cm.sum()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=CMAP_HEATMAP)
    plt.colorbar(im, ax=ax)

    ax.set_title("Matriz de confusión", fontsize=13, pad=10)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No fatal", "Pred: Fatal"])
    ax.set_yticklabels(["Real: No fatal", "Real: Fatal"])
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100 * cm[i, j] / total
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11,
            )

    output_path = OUTPUTS_DIR / "matriz_confusion.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


def plot_class_balance():
    """Distribución de la variable objetivo (fatal / no fatal)."""
    df = pd.read_csv(PROCESSED_DATA_FILE, low_memory=False)
    class_counts = df["fatal"].value_counts().sort_index()

    labels = ["No fatal (0)", "Fatal (1)"]
    values = [class_counts.get(0, 0), class_counts.get(1, 0)]
    colors = [COLOR_NO_FATAL, COLOR_FATAL]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    ax.set_title("Distribución de clases (desbalance)", fontsize=13, pad=10)
    ax.set_xlabel("Clase")
    ax.set_ylabel("Cantidad de casos")

    for bar, val in zip(bars, values):
        pct = 100 * val / sum(values)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    output_path = OUTPUTS_DIR / "desbalance_clases.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


def plot_cross_validation_results():
    """Comparación de modelos usando el F1-score promedio en validación cruzada."""
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
        classifier = str(row.get("param_classifier", "modelo"))
        if "RandomForest" in classifier:
            label = "Random Forest"
            depth = row.get("param_classifier__max_depth", "?")
            n_est = row.get("param_classifier__n_estimators", "?")
            label += f"\ndepth={depth} | n={n_est}"
        elif "LogisticRegression" in classifier:
            label = "Logistic Regression"
            c_val = row.get("param_classifier__C", "?")
            label += f"\nC={c_val}"
        else:
            label = str(classifier)[:30]
        labels.append(label)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, top_results["mean_test_f1"], color=COLOR_PRIMARY, edgecolor="white")
    ax.set_xlim(0, 1)
    ax.set_title("Comparación de modelos — F1-score en validación cruzada", fontsize=13, pad=10)
    ax.set_xlabel("F1-score promedio (CV)")
    ax.set_ylabel("Modelo e hiperparámetros")
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle="--")
    ax.invert_yaxis()

    for bar, val in zip(bars, top_results["mean_test_f1"]):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )

    output_path = OUTPUTS_DIR / "validacion_cruzada_f1.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


def plot_feature_importance():
    """Top-10 variables más importantes (solo para Random Forest)."""
    if not MODEL_FILE.exists():
        print("Aviso: no se encontró el modelo entrenado.")
        return

    model = joblib.load(MODEL_FILE)
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "feature_importances_"):
        print(
            "Aviso: el mejor modelo no expone feature_importances_ "
            "(es Logistic Regression). Se omite este gráfico."
        )
        return

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"], importance_df["importance"], color=COLOR_PRIMARY, edgecolor="white")
    ax.set_title("Importancia de variables — Top 10 (Random Forest)", fontsize=13, pad=10)
    ax.set_xlabel("Importancia (Gini)")
    ax.set_ylabel("Variable")
    ax.invert_yaxis()

    output_path = OUTPUTS_DIR / "importancia_variables.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


def plot_new_variables_vs_fatal():
    """
    Gráficos de barras apiladas para cada nueva variable categórica
    (clima, entorno, factor_humano, factor_vehiculo, tipo_auto)
    mostrando la tasa de fatalidad por categoría.
    """
    df = pd.read_csv(PROCESSED_DATA_FILE, low_memory=False)

    new_vars = {
        "clima": "Clima al momento del accidente",
        "entorno": "Entorno del accidente",
        "factor_humano": "Factor humano involucrado",
        "factor_vehiculo": "Factor vehicular involucrado",
        "tipo_auto": "Tipo de vehículo",
    }

    for col, title in new_vars.items():
        if col not in df.columns:
            print(f"Aviso: columna '{col}' no encontrada en el dataset. Se omite.")
            continue

        # Tasa de fatalidad por categoría
        summary = (
            df.groupby(col)["fatal"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "fatales", "count": "total"})
        )
        summary["tasa_fatal"] = summary["fatales"] / summary["total"]
        summary = summary.sort_values("tasa_fatal", ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(title, fontsize=13, y=1.01)

        # Panel izquierdo: conteos absolutos apilados
        no_fatal = summary["total"] - summary["fatales"]
        x = range(len(summary))
        axes[0].bar(x, no_fatal, label="No fatal", color=COLOR_NO_FATAL, edgecolor="white")
        axes[0].bar(x, summary["fatales"], bottom=no_fatal, label="Fatal", color=COLOR_FATAL, edgecolor="white")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(summary.index, rotation=30, ha="right", fontsize=9)
        axes[0].set_ylabel("Cantidad de casos")
        axes[0].set_title("Conteos absolutos")
        axes[0].legend()

        # Panel derecho: tasa de fatalidad
        axes[1].bar(x, summary["tasa_fatal"] * 100, color=COLOR_FATAL, edgecolor="white")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(summary.index, rotation=30, ha="right", fontsize=9)
        axes[1].set_ylabel("Tasa de fatalidad (%)")
        axes[1].set_title("Tasa de fatalidad por categoría")
        axes[1].set_ylim(0, summary["tasa_fatal"].max() * 100 * 1.25 + 1)

        for i, (_, row) in enumerate(summary.iterrows()):
            axes[1].text(
                i,
                row["tasa_fatal"] * 100 + 0.3,
                f"{row['tasa_fatal']*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        output_path = OUTPUTS_DIR / f"nueva_var_{col}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Gráfico guardado en: {output_path}")


def plot_roc_curve():
    """Curva ROC con AUC si el modelo soporta predict_proba."""
    if not MODEL_FILE.exists():
        print("Aviso: modelo no encontrado. Se omite curva ROC.")
        return

    import pandas as pd
    from sklearn.metrics import roc_curve, roc_auc_score
    from src.config import TEST_FEATURES_FILE, TEST_TARGET_FILE, THRESHOLD_FATAL

    model = joblib.load(MODEL_FILE)
    X_test = pd.read_csv(TEST_FEATURES_FILE)
    y_test = pd.read_csv(TEST_TARGET_FILE).squeeze()

    if not hasattr(model, "predict_proba"):
        print("Aviso: el modelo no soporta predict_proba. Se omite curva ROC.")
        return

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=COLOR_PRIMARY, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Clasificador aleatorio")

    # Marcar el umbral elegido
    idx = np.argmin(np.abs(thresholds - THRESHOLD_FATAL))
    ax.scatter(fpr[idx], tpr[idx], color=COLOR_FATAL, zorder=5, s=80,
               label=f"Umbral = {THRESHOLD_FATAL}")

    ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR / Recall)")
    ax.set_title("Curva ROC", fontsize=13, pad=10)
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])

    output_path = OUTPUTS_DIR / "curva_roc.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico guardado en: {output_path}")


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Generando visualizaciones ===\n")

    plot_metrics()
    plot_confusion_matrix()
    plot_class_balance()
    plot_cross_validation_results()
    plot_feature_importance()
    plot_new_variables_vs_fatal()   
    plot_roc_curve()                

    print("\nTodas las visualizaciones generadas correctamente.")


if __name__ == "__main__":
    main()
