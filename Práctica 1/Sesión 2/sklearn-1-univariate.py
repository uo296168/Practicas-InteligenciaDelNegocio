# %% Importaciones y rutas
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

CARPETA = Path(__file__).resolve().parent
FIGURAS = CARPETA / "figuras"


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(FIGURAS / nombre, dpi=160)
    plt.show()


# %% Lectura del CSV local
# El conjunto Boston se distribuye con la práctica. load_boston ya no existe.
datos = pd.read_csv(CARPETA / "boston.csv")
X_full = datos.drop(columns="MEDV")
y = datos["MEDV"].to_numpy()
print(X_full.columns.to_list())
print(X_full.shape)
print(y.shape)

# %% Selección descriptiva de una variable
# Se utiliza el conjunto completo porque esta primera figura es descriptiva.
# En sklearn-2.py, donde sí evaluamos, la selección se ajusta dentro de cada fold.
selector = SelectKBest(score_func=f_regression, k=1)
X = selector.fit_transform(X_full, y)
variable = X_full.columns[selector.get_support()][0]
print("Variable seleccionada:", variable)

plt.figure()
plt.scatter(X[:, 0], y, color="black", s=16)
plt.xlabel(variable)
plt.ylabel("MEDV")
guardar("sklearn-1-univariante-datos.png")

# %% Ordenación por el precio observado
# El eje horizontal es el rango de la vivienda tras ordenar y, no una variable.
orden_y = np.argsort(y)
rango = np.arange(y.size)
plt.figure()
plt.scatter(rango, y[orden_y], color="black", s=16)
plt.xlabel("Viviendas ordenadas por MEDV observado")
plt.ylabel("MEDV")
guardar("sklearn-1-univariante-precios-ordenados.png")

# %% Regresión lineal
lineal = LinearRegression()
lineal.fit(X, y)

orden_x = np.argsort(X[:, 0])
plt.figure()
plt.scatter(X[:, 0], y, color="black", s=16)
plt.plot(X[orden_x, 0], lineal.predict(X)[orden_x], color="blue", linewidth=3)
plt.xlabel(variable)
plt.ylabel("MEDV")
guardar("sklearn-1-univariante-lineal.png")

plt.figure()
plt.plot(rango, y[orden_y], color="black", label="Observado")
plt.scatter(rango, lineal.predict(X)[orden_y], color="blue", s=15, label="Lineal")
plt.xlabel("Viviendas ordenadas por MEDV observado")
plt.ylabel("MEDV")
plt.legend()
guardar("sklearn-1-univariante-lineal-orden-y.png")

# %% Regresión SVR
svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=10, epsilon=1))
svr.fit(X, y)

plt.figure()
plt.scatter(X[:, 0], y, color="black", s=16)
plt.scatter(X[:, 0], svr.predict(X), color="blue", s=15)
plt.xlabel(variable)
plt.ylabel("MEDV")
guardar("sklearn-1-univariante-svr.png")

plt.figure()
plt.plot(rango, y[orden_y], color="black", label="Observado")
plt.scatter(rango, svr.predict(X)[orden_y], color="blue", s=15, label="SVR")
plt.xlabel("Viviendas ordenadas por MEDV observado")
plt.ylabel("MEDV")
plt.legend()
guardar("sklearn-1-univariante-svr-orden-y.png")

# %% Bosque aleatorio
bosque = RandomForestRegressor(n_estimators=300, random_state=42)
bosque.fit(X, y)

plt.figure()
plt.scatter(X[:, 0], y, color="black", s=16)
plt.scatter(X[:, 0], bosque.predict(X), color="blue", s=15)
plt.xlabel(variable)
plt.ylabel("MEDV")
guardar("sklearn-1-univariante-bosque.png")

plt.figure()
plt.plot(rango, y[orden_y], color="black", label="Observado")
plt.scatter(rango, bosque.predict(X)[orden_y], color="blue", s=15, label="Bosque")
plt.xlabel("Viviendas ordenadas por MEDV observado")
plt.ylabel("MEDV")
plt.legend()
guardar("sklearn-1-univariante-bosque-orden-y.png")
