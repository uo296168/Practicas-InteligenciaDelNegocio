# %% DataFrames de ejemplo
import pandas as pd

df1 = pd.DataFrame(
    {
        "column_a": [1, 2, 3, 4],
        "column_b": ["a", "b", "c", "d"],
        "column_c": [True, True, False, True],
    }
)
df2 = pd.DataFrame(
    {
        "column_a": [1, 2, 9, 10],
        "column_b": ["a", "k", "l", "m"],
        "column_c": [False, False, False, True],
    }
)
print(df1)
print(df2)

# %% Concatenación por filas
print(pd.concat([df1, df2]))
print(pd.concat([df1, df2], ignore_index=True))

# %% Concatenación por columnas usando los índices actuales
print(pd.concat([df1, df2], axis=1))

# %% Efecto de índices no coincidentes
df2_desplazado = df2.copy()
df2_desplazado.index = [3, 4, 5, 6]
print(pd.concat([df1, df2_desplazado], axis=1, join="inner"))
print(pd.concat([df1, df2_desplazado], axis=1, join="outer"))
