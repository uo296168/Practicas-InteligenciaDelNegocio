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

# %% Selección mediante posiciones: iloc
print(df.iloc[1])
print(df.iloc[0, 1])
print(df.iloc[:, 1])
print(df.iloc[:2, 1])

# %% Selección mediante etiquetas: loc
# En un índice entero, el extremo final de loc está incluido.
print(df.loc[:2, "ciudad"])
print(df.loc[:2, :"ciudad"])
