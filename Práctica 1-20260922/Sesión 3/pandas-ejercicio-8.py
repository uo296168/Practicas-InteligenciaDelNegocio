# %% Importaciones y lectura
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.inspection import permutation_importance

CARPETA = Path(__file__).resolve().parent
df = pd.read_excel(CARPETA / "Churn_Modelling_sintetico_NANs.xlsx")

# %% Primer diagnóstico
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isna().sum())

# %% Ejercicio 8
# 1. Elimine las filas con valores perdidos.
df = df.dropna()
# 2. Seleccione X con las columnas que vea razonables.
X = df[
    [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
    ]
]
# 3. Seleccione y = EstimatedSalary.

y = df["EstimatedSalary"]

# 4. Compare regresión lineal, SVR y bosque aleatorio mediante validación cruzada de 10 folds y error cuadrático medio.

modelos = {
    "Regresión lineal": LinearRegression(),

    "SVR": make_pipeline(
        StandardScaler(),
        SVR()
    ),

    "Bosque aleatorio": RandomForestRegressor(
        random_state=42
    )
}

particiones = KFold(n_splits=10,
                    shuffle=True, 
                    random_state=42)

for nombre, modelo in modelos.items():

    puntuaciones = cross_val_score(
        modelo,
        X,
        y,
        cv = particiones ,
        scoring="neg_mean_squared_error"
    )

    mse = -puntuaciones.mean()

    print(nombre, "MSE:", mse)

# %% 5. Opcional: prediga Exited mediante tres clasificadores y compare aciertos. Implementar la matriz de confusion

X_clas = df[
    [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
    ]
]

y_clas = df["Exited"]


clasificadores = {

    "Regresión logística": make_pipeline(
        StandardScaler(),
        LogisticRegression()
    ),

    "SVC": make_pipeline(
        StandardScaler(),
        SVC()
    ),

    "Bosque aleatorio": RandomForestClassifier(
        random_state=42
    )
}


for nombre, modelo in clasificadores.items():

    puntuaciones = cross_val_score(
        modelo,
        X_clas,
        y_clas,
        cv=particiones,
        scoring="accuracy"
    )

    accuracy = puntuaciones.mean()

    print(nombre, "Accuracy:", accuracy)
    
# %% 6. Opcional: sustituya la eliminación por diferentes métodos de imputación.

# Implementar la matriz de confusión

# Metodo de imputación: rellenar con la media de cada columna numérica.

df_imputado = df.copy()

columnas_numericas = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
]

for columna in columnas_numericas:
    df_imputado[columna] = df_imputado[columna].fillna(
        df_imputado[columna].mean()
    )

print(df_imputado.isna().sum())

X = df_imputado[
    [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
    ]
]

y = df_imputado["EstimatedSalary"]

for nombre, modelo in modelos.items():

    puntuaciones = cross_val_score(
        modelo,
        X,
        y,
        cv=particiones,
        scoring="neg_mean_squared_error"
    )

    mse = -puntuaciones.mean()

    print(nombre, "MSE (imputado):", mse)
    

# 7. Discuta qué variables influyen en cada predicción. 

# %% Importancia de variables - Regresión lineal

modelo_lineal = LinearRegression()
modelo_lineal.fit(X, y)

importancias_lineal = pd.Series(
    modelo_lineal.coef_,
    index=X.columns
)

print("\nCoeficientes de regresión lineal:")
print(importancias_lineal.sort_values(key=abs, ascending=False))

# %% Importancia de variables - Random Forest

modelo_bosque = RandomForestRegressor(
    random_state=42
)

modelo_bosque.fit(X, y)

importancias_bosque = pd.Series(
    modelo_bosque.feature_importances_,
    index=X.columns
)

print("\nImportancia de variables en Random Forest:")
print(importancias_bosque.sort_values(ascending=False))

# %% Importancia de variables - SVR

modelo_svr = make_pipeline(
    StandardScaler(),
    SVR()
)

modelo_svr.fit(X, y)

resultado = permutation_importance(
    modelo_svr,
    X,
    y,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_squared_error"
)

importancias_svr = pd.Series(
    resultado.importances_mean,
    index=X.columns
)

print("\nImportancia de variables en SVR:")
print(importancias_svr.sort_values(ascending=False))


