import pandas as pd
from src.config import RAW_DATA_FILE, PROCESSED_DATA_FILE
from src.db import insert_processed_data


def load_data():
    """
    Carga el archivo Excel original.
    """
    df = pd.read_excel(RAW_DATA_FILE)
    return df


def preprocess_data(df):
    """
    Limpia y transforma el dataset para dejarlo listo para el modelado.
    """

    # Reemplazar "SD" por valores nulos
    df = df.replace("SD", pd.NA)

    print("Columnas originales:")
    print(df.columns.tolist())

    # Convertir columnas de fecha
    if "fecha_siniestro" in df.columns:
        df["fecha_siniestro"] = pd.to_datetime(
            df["fecha_siniestro"],
            errors="coerce"
        )

    if "fecha_fallecimiento_victima" in df.columns:
        df["fecha_fallecimiento_victima"] = pd.to_datetime(
            df["fecha_fallecimiento_victima"],
            errors="coerce"
        )
    else:
        raise ValueError(
            "No se encontró la columna 'fecha_fallecimiento_victima'."
        )

    # Crear variable objetivo
    df["fatal"] = df["fecha_fallecimiento_victima"].notna().astype(int)

    # Extraer variables desde fecha_siniestro
    if "fecha_siniestro" in df.columns:
        df["anio"] = df["fecha_siniestro"].dt.year
        df["mes"] = df["fecha_siniestro"].dt.month
        df["dia_semana"] = df["fecha_siniestro"].dt.dayofweek

    # Convertir edad a numérica por si viene como texto
    if "edad_victima" in df.columns:
        df["edad_victima"] = pd.to_numeric(
            df["edad_victima"],
            errors="coerce"
        )

    # Seleccionar columnas finales para el modelo
    columnas_modelo = [
        "anio",
        "edad_victima",
        "sexo_victima",
        "modo_desplazamiento_victima",
        "rol_victima",
        "mes",
        "dia_semana",
        "fatal"
    ]

    # Quedarse solo con las columnas que existen
    columnas_existentes = [col for col in columnas_modelo if col in df.columns]
    df = df[columnas_existentes].copy()

    return df


def save_processed_data(df):
    """
    Guarda el dataset procesado en CSV.
    """
    PROCESSED_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)


def save_to_mongodb(df):
    """
    Inserta el dataset procesado en MongoDB.
    Cada fila se guarda como un documento.
    """
    records = df.to_dict(orient="records")
    if records:
        insert_processed_data(records)


if __name__ == "__main__":
    df = load_data()
    df_clean = preprocess_data(df)

    print("\nPrimeras filas del dataset procesado:")
    print(df_clean.head())

    save_processed_data(df_clean)
    save_to_mongodb(df_clean)

    print("\nPreprocesamiento finalizado.")
    print(f"Archivo procesado guardado en: {PROCESSED_DATA_FILE}")
    print("Si MongoDB estaba disponible, los datos fueron insertados en la colección datos_entrada.")