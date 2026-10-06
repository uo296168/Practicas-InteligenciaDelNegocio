# %% DataFrame con distintas representaciones de valores perdidos
import numpy as np
import pandas as pd

df = pd.DataFrame(
    {
        "column_a": [1, 2, 4, 4, np.nan, np.nan, 6],
        "column_b": [1.2, 1.4, np.nan, 6.2, None, 1.1, 4.3],
        "column_c": ["a", "?", "c", "d", "--", np.nan, "d"],
        "column_d": [True, True, np.nan, None, False, True, False],
    }
)
df["column_e"] = pd.Series([1, 2, np.nan, 4, np.nan, 5], dtype="Int64")
print(df)

# %% Detección y recuento
print(df.isna())
print(df.notna())
print(df.isna().any())
print(df.isna().sum())

# %% Normalización de símbolos que representan ausencia
normalizado = df.replace(["?", "--"], np.nan)
print(normalizado)

# %% Eliminación: cada operación parte del mismo DataFrame
print(normalizado.dropna(axis=0, how="all"))
print(normalizado.dropna(axis=1, how="any"))
print(normalizado.dropna(axis=0, how="any"))

# %% Imputación mediante una constante o un estadístico
print(normalizado.fillna(25))
con_media = normalizado.copy()
con_media["column_a"] = con_media["column_a"].fillna(con_media["column_a"].mean())
print(con_media)

# %% Propagación del valor anterior o siguiente
# Solo tiene sentido cuando el orden de las filas es relevante.
print(normalizado.ffill())
print(normalizado.bfill())
print(normalizado.ffill(limit=1))
