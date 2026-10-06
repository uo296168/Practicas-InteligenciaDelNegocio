# %% Importaciones y rutas
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent

# %% Lectura del conjunto completo desde CSV
df = pd.read_csv(CARPETA / "Churn_Modelling_sintetico.csv")
print(df.head())

# %% Lectura selectiva de columnas y filas
columnas = ["CustomerId", "CreditScore", "NumOfProducts"]
seleccion = pd.read_csv(
    CARPETA / "Churn_Modelling_sintetico.csv", usecols=columnas, nrows=500
)
print(seleccion.head())
print(seleccion.shape)

# %% Lectura y escritura de Excel
df_excel = pd.read_excel(CARPETA / "Churn_Modelling_sintetico.xlsx")
df_excel.head(100).to_excel(CARPETA / "miarchivo.xlsx", index=False)
print(df_excel.head())

# %% Creación de un DataFrame desde un diccionario
# Cada clave debe ser distinta: una clave repetida sobrescribe la anterior.
rng = np.random.default_rng(42)
ejemplo = pd.DataFrame(
    {
        "a": rng.random(10),
        "b": rng.integers(10, size=10),
        "c": [True, True, True, False, False, np.nan, np.nan, False, True, True],
        "ciudad": [
            "London", "Paris", "New York", "Istanbul", "Liverpool",
            "Berlin", np.nan, "Madrid", "Rome", np.nan,
        ],
        "d": [3, 4, 5, 1, 5, 2, 2, np.nan, np.nan, 0],
        "e": [1, 4, 5, 3, 3, 3, 3, 8, 8, 4],
    }
)
print(ejemplo)
