from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_target_distribution(df: pd.DataFrame, output_path: str | Path) -> None:
    counts = df["fatal"].value_counts().sort_index()
    labels = ["No fatal", "Fatal"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts.values)
    plt.title("Distribución de la variable objetivo")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de casos")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_metrics_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> None:
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    plot_df = metrics_df.set_index("model_name")[metrics_to_plot]

    plt.figure(figsize=(9, 5))
    plot_df.T.plot(kind="bar")
    plt.title("Comparación de métricas por modelo")
    plt.xlabel("Métrica")
    plt.ylabel("Valor")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
