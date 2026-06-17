from pathlib import Path
import joblib
import pandas as pd
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
X_TEST_PATH = BASE_DIR / "outputs" / "X_test.csv"
Y_TEST_PATH = BASE_DIR / "outputs" / "y_test.csv"

model = joblib.load(MODEL_PATH)

X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

# Unir X_test con el valor real para poder elegir casos fatales y no fatales
df_test = X_test.copy()
df_test["fatal_real"] = y_test.values

# Tomar algunos casos reales NO vistos por el modelo
casos_no_fatales = df_test[df_test["fatal_real"] == 0].sample(
    n=5,
    random_state=42
)

casos_fatales = df_test[df_test["fatal_real"] == 1].sample(
    n=5,
    random_state=42
)

casos = pd.concat([casos_no_fatales, casos_fatales])


FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


for i, fila in casos.iterrows():
    X_nuevo = pd.DataFrame([fila[FEATURES]])
    valor_real = fila["fatal_real"]

    prediccion = model.predict(X_nuevo)[0]
    probabilidad_fatal = model.predict_proba(X_nuevo)[0][1]

    print("=" * 80)
    print(f"Caso de prueba ID original: {i}")
    print(X_nuevo)

    print(f"Valor real: {'FATAL' if valor_real == 1 else 'NO FATAL'}")
    print(f"Predicción modelo: {'FATAL' if prediccion == 1 else 'NO FATAL'}")
    print(f"Probabilidad estimada de fatalidad: {probabilidad_fatal:.4f}")

    if prediccion == valor_real:
        print("Resultado: coincide con el valor real")
    else:
        print("Resultado: NO coincide con el valor real")