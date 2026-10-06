# %% DataFrame de ejemplo
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
df = pd.DataFrame(
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
print(df)

# %% Descripción estadística de los datos numéricos
# Los valores NaN no se incluyen en count ni en los estadísticos.
print(df.describe())

# %% Acceso a una columna y recuento de valores
print(df["c"])
print(df["c"].value_counts(dropna=False))

# %% Tipos de datos inferidos
print(df.dtypes)
