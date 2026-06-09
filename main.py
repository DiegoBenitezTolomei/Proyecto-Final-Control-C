from src.preprocessing import load_data, preprocess_data, save_processed_data, save_to_mongodb
from src.train import train_models
from src.evaluate import evaluate_model
from src.visualize import main as generate_visualizations


def main():
    print("1/4 - Preprocesamiento de datos")
    df_raw = load_data()
    df_clean = preprocess_data(df_raw)
    save_processed_data(df_clean)
    save_to_mongodb(df_clean)

    print("\n2/4 - Entrenamiento y validación cruzada")
    train_models()

    print("\n3/4 - Evaluación sobre conjunto de prueba")
    evaluate_model()

    print("\n4/4 - Generación de visualizaciones")
    generate_visualizations()

    print("\nPipeline completo finalizado.")


if __name__ == "__main__":
    main()
