# %% Importaciones y datos
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

CARPETA = Path(__file__).resolve().parent

datos = pd.read_csv(CARPETA / "boston.csv")
X = datos.drop(columns="MEDV")
y = datos["MEDV"].to_numpy()


# %% Modelos equivalentes a los de sklearn-1

# SelectKBest se ajusta dentro de cada fold para no utilizar el conjunto de
# prueba al elegir la variable. SVR incluye además el escalado necesario.

modelos = {
    "Lineal": make_pipeline(
        SelectKBest(score_func=f_regression, k=1),
        LinearRegression()
    ),

    "SVR": make_pipeline(
        SelectKBest(score_func=f_regression, k=1),
        StandardScaler(),
        SVR(kernel="rbf", C=10, epsilon=1),
    ),

    "Bosque": make_pipeline(
        SelectKBest(score_func=f_regression, k=1),
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),
    ),
}


# %% Validación cruzada con las mismas particiones

particiones = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

predicciones = {}
filas = []

for nombre, modelo in modelos.items():

    scores = cross_val_score(
        modelo,
        X,
        y,
        scoring="neg_mean_squared_error",
        cv=particiones
    )

    prediccion = cross_val_predict(
        modelo,
        X,
        y,
        cv=particiones
    )

    predicciones[nombre] = prediccion

    filas.append(
        {
            "modelo": nombre,
            "MSE: media de folds": -scores.mean(),
            "MSE: predicciones reunidas": mean_squared_error(
                y,
                prediccion
            ),
        }
    )

resultados = pd.DataFrame(filas).set_index("modelo")

print(resultados.round(3))


# %% Predicciones de validación cruzada ordenadas por el precio observado

orden_y = np.argsort(y)
rango = np.arange(y.size)

fig, ejes = plt.subplots(
    1,
    3,
    figsize=(13, 4),
    sharey=True
)

for eje, (nombre, prediccion) in zip(ejes, predicciones.items()):

    eje.plot(
        rango,
        y[orden_y],
        color="black",
        linewidth=1.5,
        label="Observado"
    )

    eje.scatter(
        rango,
        prediccion[orden_y],
        s=9,
        alpha=0.75,
        label=nombre
    )

    eje.set_title(nombre)
    eje.set_xlabel("Rango por MEDV observado")

ejes[0].set_ylabel("MEDV")
ejes[0].legend()

fig.suptitle("Predicciones de validación cruzada")

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()
