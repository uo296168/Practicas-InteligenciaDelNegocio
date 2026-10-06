# %% Importaciones y rutas
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from klearn.feature_selection import SelectorBest, f_regression

CARPETA = Path(__file__).resolve().parent
FIGURAS = CARPETA / "figuras"


def guardar(nombre, rect=None):
    plt.tight_layout(rect=rect)
    plt.savefig(FIGURAS / nombre, dpi=160)
    plt.show()


# %% Lectura y modelos con todas las variables
datos = pd.read_csv(CARPETA / "boston.csv")
X = datos.drop(columns="MEDV")
y = datos["MEDV"].to_numpy()
print(X.shape)
print(y.shape)

k = 5

modelos = {
    "Lineal": make_pipeline(SelectorBest(score_func_f_regression, k=k), LinearRegression()),
    "SVR": make_pipeline(SelectorBest(score_func_f_regression, k=k), StandardScaler(), SVR(kernel="rbf", C=10, epsilon=1)),
    "Bosque": make_pipeline(SelectorBest(score_func_f_regression, k=k), RandomForestRegressor(n_estimators=300, random_state=42)),
}
predicciones = {}
for nombre, modelo in modelos.items():
    modelo.fit(X, y)
    predicciones[nombre] = modelo.predict(X)

# %% Ordenación por el precio observado
# Con 13 variables de entrada no existe un único eje X para un scatter 2D.
# Sustituimos ese eje por el rango de las viviendas al ordenar MEDV.
orden_y = np.argsort(y)
rango = np.arange(y.size)

plt.figure()
plt.scatter(rango, y[orden_y], color="black", s=16)
plt.xlabel("Viviendas ordenadas por MEDV observado")
plt.ylabel("MEDV")
guardar("sklearn-1-multivariante-objetivo-ordenado.png")

for nombre, prediccion in predicciones.items():
    plt.figure()
    plt.plot(rango, y[orden_y], color="black", label="Observado")
    plt.scatter(rango, prediccion[orden_y], s=15, label=nombre)
    plt.xlabel("Viviendas ordenadas por MEDV observado")
    plt.ylabel("MEDV")
    plt.legend()
    guardar(f"sklearn-1-multivariante-{nombre.lower()}-orden-y.png")

# %% Comparación conjunta: viviendas baratas y caras según el valor observado
fig, ejes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for eje, (nombre, prediccion) in zip(ejes, predicciones.items()):
    eje.plot(rango, y[orden_y], color="black", linewidth=1.5, label="Observado")
    eje.scatter(rango, prediccion[orden_y], s=9, alpha=0.75, label=nombre)
    eje.set_title(nombre)
    eje.set_xlabel("Rango por MEDV observado")
ejes[0].set_ylabel("MEDV")
ejes[0].legend()
fig.suptitle("Predicciones al ordenar las viviendas por el precio observado")
guardar("sklearn-1-multivariante-comparacion-orden-y.png", rect=[0, 0, 1, 0.92])
