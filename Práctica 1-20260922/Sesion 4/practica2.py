# %% [markdown]
# # Práctica 2. Preparación de datos para una campaña de retención
#
# Un banco pierde cada año en torno al 19 % de sus clientes. Su centro de llamadas puede contactar con el 15 %
# de la cartera para ofrecer una bonificación, y la dirección comercial quiere saber **a quién llamar**.
#
# - Llamar a un cliente que iba a marcharse aporta en promedio **140 €**: tres de cada diez se quedan (600 €
#   de margen) y la llamada cuesta 40 €.
# - Llamar a uno que se habría quedado cuesta **40 €**.
#
# El modelo ordena a los clientes por riesgo y se llama al 15 % superior. Sobre esa lista se mide:
#
# | Métrica | Lectura para el banco |
# |---|---|
# | Precisión | % de llamadas que llegan a un cliente que iba a marcharse |
# | Exhaustividad | % de los abandonos del año que reciben una llamada |
# | Beneficio | € por cada 1.000 clientes: 140 por llamada útil − 40 por llamada inútil |
#
# La campaña solo es rentable si la precisión supera el 22 % (40 / 180): llamar al azar pierde dinero.
#
# **Normas.** Todo lo que se aprende de los datos (escalas, medianas, variables elegidas…) se ajusta dentro de
# un `Pipeline` y solo con entrenamiento. Las alternativas se comparan por validación cruzada sobre el
# entrenamiento; el test se usa una sola vez, al final. Entre alternativas con resultados parecidos se elige
# la más sencilla.
#
# Ejecuta el archivo celda a celda. Las tareas están marcadas con **TAREA**; escribe tus respuestas en un documento
# aparte, junto con los comentarios que tengas, y entrega tanto el documento como este notebook completado cuando
# abramos la entrega de las prácticas P1, P2, P3.
#
# No basta con entregar solo el notebook sin el documento.


# %% [markdown]
# ## Datos
#
# `clientes_banco.csv`: 6.000 clientes a 30/09/2025; `abandono` = 1 si el cliente se marchó en los doce meses
# siguientes.
#
# | Columna | Contenido |
# |---|---|
# | `edad`, `anio_nacimiento` | edad a 30/09/2025 y año de nacimiento |
# | `antiguedad`, `num_productos` | años como cliente y productos contratados |
# | `cliente_activo` | 1 si `transacciones_mes` ≥ 10 o `accesos_app_mes` ≥ 6 (regla del banco) |
# | `ingresos_mensuales` | nómina domiciliada (€); vacío si no la domicilia en el banco |
# | `salario_estimado` | estimación anual de un proveedor (€); cuesta 0,20 € por cliente y año |
# | `puntuacion_credito` | puntuación de una agencia (350–850); vacía si no hay expediente; 0,50 € por cliente y año |
# | `saldo_hace_3m`, `saldo_actual` | saldo a 30/06 y a 30/09 (€); el primero falta en los registros dañados por una migración |
# | `transacciones_mes`, `accesos_app_mes` | movimientos y accesos a la app en septiembre |
# | `reclamaciones_12m` | reclamaciones en el último año |
#
# `saldos_recuperados.csv` contiene los saldos perdidos en la migración, recuperados de una copia de
# seguridad. No se usa para entrenar: solo mide el error de imputación (sección 5).


# %% Imports y datos
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler, StandardScaler)
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import (CondensedNearestNeighbour, EditedNearestNeighbours,
                                     RandomUnderSampler, TomekLinks)

from utilidades import criterio_beneficio, error_saldo, evaluar_regla, indicadores, mostrar, validar

warnings.filterwarnings("ignore")
set_config(transform_output="pandas")  # los pasos del pipeline conservan los nombres de columna
CARPETA = Path(__file__).resolve().parent

clientes = pd.read_csv(CARPETA / "clientes_banco.csv")
recuperados = pd.read_csv(CARPETA / "saldos_recuperados.csv")
y = clientes["abandono"]
X = clientes.drop(columns=["abandono", "id_cliente"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
print("Entrenamiento:", X_train.shape, f"| abandono: {y_train.mean():.1%}")


# %% [markdown]
# ## 1. Modelos de referencia
#
# Un modelo solo se justifica si mejora lo que el banco ya hace sin él. Se comparan la **regla actual**
# (llamar primero a quien tiene más reclamaciones y después a los inactivos), una lista **al azar** y un
# **modelo inicial**: regresión logística con todas las columnas estandarizadas y los vacíos imputados con la
# mediana.
#
# En las tablas, `±` es la desviación típica del beneficio entre los cinco pliegues y `Mejora` cuenta en
# cuántos pliegues el método supera en beneficio a la primera fila. Como todos usan los mismos pliegues, 5/5
# indica una ventaja consistente; 2/5 o 3/5, una diferencia que puede deberse al azar.


# %% 1. Referencias
def regla_actual(D):
    return D["reclamaciones_12m"] + (1 - D["cliente_activo"])


def azar(D):
    return np.random.default_rng(0).random(len(D))


def modelo_lr(escalado=None, imputador=None):
    # escalado -> imputación -> regresión logística
    return Pipeline([
        ("escalado", escalado if escalado is not None else StandardScaler()),
        ("imputacion", imputador if imputador is not None else SimpleImputer(strategy="median")),
        ("modelo", LogisticRegression(max_iter=1000)),
    ])


mostrar([
    evaluar_regla("Regla actual", regla_actual, X_train, y_train),
    evaluar_regla("Azar", azar, X_train, y_train),
    validar("Modelo inicial", modelo_lr(), X_train, y_train),
])


# %% [markdown]
# ## TAREA 1
# Escribe una regla sin aprendizaje que sume a la regla actual el doble de la retirada de saldo del
# trimestre, retirada = 1 - (saldo_actual + 100) / (saldo_hace_3m + 100), con los valores negativos y vacíos
# sustituidos por 0. Compárala con la regla actual y con el modelo inicial.

# RESPUESTA (¿supera al modelo inicial? ¿qué implica para la decisión de usar un modelo?):
#


# %% [markdown]
# ## 2. Redundancia (clase 4)
#
# Al unir sistemas, la misma información puede llegar en varias columnas. Una correlación alta señala pares
# que conviene revisar; para retirar una columna hay que explicar qué información repite.


# %% 2. Pares correlacionados y efecto de retirar columnas
corr = X_train.corr()
pares = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
print(pares.sort_values(key=np.abs, ascending=False).head(4).round(3), "\n")

mostrar([
    validar("Todas las columnas", modelo_lr(), X_train, y_train),
    validar("Sin anio_nacimiento", modelo_lr(), X_train.drop(columns=["anio_nacimiento"]), y_train),
    validar("Sin anio_nacimiento ni saldo_hace_3m", modelo_lr(),
            X_train.drop(columns=["anio_nacimiento", "saldo_hace_3m"]), y_train),
])


# %% [markdown]
# ## TAREA 2
# Comprueba que cliente_activo coincide siempre con su regla del diccionario 
# (vuelve a la celda markdown con título "Datos" para ver esa "regla del banco").

# RESPUESTA (¿qué columnas retirarías y por qué? ¿por qué no saldo_hace_3m, pese a su correlación con
# saldo_actual? ¿por qué la correlación no revela que cliente_activo es redundante?):
#


# %% Columnas que se conservan
X_train, X_test = X_train.drop(columns=["anio_nacimiento"]), X_test.drop(columns=["anio_nacimiento"])


# %% [markdown]
# ## 3. Escalado (clase 4)
#
# La dirección quiere justificar cada llamada con «clientes parecidos que se marcharon»: un kNN. La distancia
# euclídea suma diferencias en las unidades de cada columna; sin escalar, dominan las de mayor varianza.
# Elegir un escalado equivale a decidir qué es «parecido».


# %% 3. kNN con distintos escalados
varianza = X_train.var()
print((100 * varianza / varianza.sum()).sort_values(ascending=False).head(3).round(1), "\n")  # % de la varianza

ESCALADOS = [("sin escalar", "passthrough"), ("StandardScaler", StandardScaler()),
             ("MinMaxScaler", MinMaxScaler()), ("RobustScaler", RobustScaler())]


def modelo_knn(escalado):
    return Pipeline([("escalado", escalado), ("imputacion", SimpleImputer(strategy="median")),
                     ("modelo", KNeighborsClassifier(n_neighbors=25))])


mostrar([validar(f"kNN · {nombre}", modelo_knn(e), X_train, y_train) for nombre, e in ESCALADOS])


# %% [markdown]
# ## TAREA 3
# Repite la comparación con la regresión logística (modelo_lr) en lugar del kNN.

# RESPUESTA (¿qué escalado eliges para el kNN y qué considera «parecido»? ¿por qué el escalado influye menos
# en la regresión logística?):
#


# %% [markdown]
# ## 4. Transformaciones y variables derivadas (clase 5)
#
# Según el área comercial, lo que anticipa un abandono no es tener poco saldo, sino haberlo retirado: pasar de
# 50.000 € a 30.000 € equivale a pasar de 5.000 € a 3.000 €. La señal es relativa, y hay dos formas de
# ofrecérsela al modelo:
#
# - una **transformación estimada** de cada columna (Yeo–Johnson, cuantiles), que cambia su distribución;
# - una **variable derivada** que exprese la hipótesis: `var_saldo` = log((saldo_actual + 100) /
#   (saldo_hace_3m + 100)). Es una regla fija: no aprende nada de los datos.
#
# Como referencia se incluye un modelo complejo, *gradient boosting*, que no necesita transformaciones.


# %% 4. Representaciones
def anadir_derivadas(D):
    D = D.copy()
    # TAREA 4: añade aquí la columna var_saldo
    return D


def modelo_derivadas(imputador=None):
    # variable derivada -> estandarización -> imputación -> regresión logística
    return Pipeline([("derivadas", FunctionTransformer(anadir_derivadas))] + modelo_lr(imputador=imputador).steps)


def boosting(derivadas):
    pasos = [("derivadas", FunctionTransformer(anadir_derivadas))] if derivadas else []
    return Pipeline(pasos + [("imputacion", SimpleImputer(strategy="median")),
                             ("modelo", HistGradientBoostingClassifier(random_state=0))])


mostrar([
    validar("Estandarización", modelo_lr(), X_train, y_train),
    validar("Yeo–Johnson", modelo_lr(PowerTransformer()), X_train, y_train),
    validar("Cuantiles", modelo_lr(QuantileTransformer(output_distribution="normal", random_state=0)), X_train, y_train),
    validar("Variable derivada + estandarización", modelo_derivadas(), X_train, y_train),
    validar("Gradient boosting", boosting(False), X_train, y_train),
    validar("Gradient boosting + variable derivada", boosting(True), X_train, y_train),
])


# %% [markdown]
# ## TAREA 4
# Completa la función "anadir_derivadas" con var_saldo y vuelve a ejecutar la celda anterior.

# RESPUESTA (¿qué representación eliges para la regresión logística? ¿qué aprende de los datos cada
# transformación? ¿a qué modelos beneficia la variable derivada?):
#


# %% [markdown]
# ## 5. Valores perdidos (clases 6 y 7)
#
# El modelo debe puntuar a todos los clientes: quien no tiene puntuación no puede recibir una llamada. Antes
# de imputar se diagnostica cuánto falta y si la ausencia se relaciona con otras variables o con el abandono.


# %% 5. Diagnóstico
ausentes = ["ingresos_mensuales", "puntuacion_credito", "saldo_hace_3m"]
print(pd.DataFrame({
    "% sin dato": 100 * X_train[ausentes].isna().mean(),
    "% abandono si falta": [100 * y_train[X_train[c].isna()].mean() for c in ausentes],
    "% abandono si consta": [100 * y_train[X_train[c].notna()].mean() for c in ausentes],
}).round(1), "\n")
tramo = pd.cut(X_train["edad"], [17, 30, 50, 90])
print((100 * X_train[ausentes].isna().groupby(tramo, observed=True).mean()).round(1))  # % sin dato por edad


# %% [markdown]
# Se comparan, con la representación de la sección 4, varios tratamientos: imputar la mediana; la mediana con
# un **indicador de ausencia** (una columna 0/1 que avisa de que el dato faltaba); KNN, que imputa con
# clientes parecidos; imputación iterativa, que predice cada columna a partir de las demás; y eliminar a los
# clientes incompletos, que entonces no se puntúan. `Error saldo` es la mediana del error, en euros, al
# reconstruir `saldo_hace_3m`.


# %% 5. Tratamientos de los vacíos
ids_train = clientes.loc[X_train.index, "id_cliente"]
filas = []
for nombre, imputador in [
    ("Mediana", SimpleImputer(strategy="median")),
    ("Mediana + indicador", SimpleImputer(strategy="median", add_indicator=True)),
    ("KNN", KNNImputer(n_neighbors=10, weights="distance")),
    ("Iterativa", IterativeImputer(max_iter=20, tol=1e-2, random_state=0)),
]:
    fila = validar(nombre, modelo_derivadas(imputador), X_train, y_train)
    filas.append({**fila, **error_saldo(imputador, X_train, ids_train, recuperados)})
filas.append(validar("Eliminar incompletos", modelo_derivadas(), X_train, y_train, solo_completos=True))
mostrar(filas)


# %% [markdown]
# ## TAREA 5
# Interpreta el diagnóstico y la comparación anteriores.

# RESPUESTA (clasifica la ausencia de cada columna como MCAR, MAR o MNAR con la evidencia del diagnóstico;
# ¿qué tratamiento eliges? ¿coincide el que mejor reconstruye el saldo con el que produce mejor campaña? ¿por
# qué no se puede eliminar a los clientes incompletos?):
#


# %% Preparación elegida
def preparacion(derivadas=True):
    # variable derivada -> estandarización -> mediana con indicador de ausencia
    pasos = [("derivadas", FunctionTransformer(anadir_derivadas))] if derivadas else []
    return pasos + [("escalado", StandardScaler()),
                    ("imputacion", SimpleImputer(strategy="median", add_indicator=True))]


# %% [markdown]
# ## 6. Selección de características (clase 8)
#
# Cada variable cuesta: dos se compran (0,70 € por cliente y año entre ambas) y todas hay que mantenerlas y
# explicarlas. Se comparan las tres familias de selección: **filtro** (puntúa cada variable por separado:
# ANOVA), **wrapper** (evalúa subconjuntos con el modelo: RFECV, aquí con el beneficio como criterio) y
# **embebido** (la penalización L1 anula coeficientes durante el ajuste). El indicador de ausencia de un dato
# comprado también obliga a comprarlo.


# %% 6. Selección de características
COSTE_DATOS = {"puntuacion_credito": 0.50, "salario_estimado": 0.20}  # € por cliente y año

filas_sel = []
for nombre, selector, derivadas in [
    ("Sin selección", "passthrough", True),
    ("Filtro ANOVA, k=8 (sin var_saldo)", SelectKBest(f_classif, k=8), False),
    ("Filtro ANOVA, k=8", SelectKBest(f_classif, k=8), True),
    ("Wrapper RFECV", RFECV(LogisticRegression(max_iter=1000), cv=5, scoring=criterio_beneficio), True),
    ("Embebido L1", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", C=0.05)), True),
]:
    pipe = Pipeline(preparacion(derivadas) + [("seleccion", selector), ("modelo", LogisticRegression(max_iter=1000))])
    fila = validar(nombre, pipe, X_train, y_train)
    usadas = list(pipe.fit(X_train, y_train).named_steps["modelo"].feature_names_in_)
    coste = 1000 * sum(c for dato, c in COSTE_DATOS.items() if any(dato in v for v in usadas))
    filas_sel.append({**fila, "Nº variables": len(usadas), "Coste datos (€/1.000)": coste,
                      "Beneficio neto (€/1.000)": fila["Beneficio (€/1.000)"] - coste})
mostrar(filas_sel)


# %% [markdown]
# ## TAREA 6
# Evalúa el modelo sin las dos variables compradas (sin seleccionar el resto) y calcula el ahorro
# anual en una cartera de 250.000 clientes.

# RESPUESTA (¿puede el banco dejar de comprarlas? ¿qué método de selección eliges y por qué? ¿por qué el
# filtro descarta saldo_hace_3m cuando no existe var_saldo?):
#


# %% [markdown]
# ## 7. Selección de instancias (clase 8)
#
# El servicio de vecinos de la sección 3 compara cada cliente con todo el histórico, que crece cada año.
# **CNN** condensa: conserva los casos necesarios para mantener la frontera entre clases. **ENN** y **Tomek**
# editan: eliminan casos que contradicen a sus vecinos. Solo se reduce el entrenamiento (por defecto, solo los
# clientes que no abandonan); la validación conserva a todos. Se usan tres pliegues porque CNN es lento.


# %% 7. Reducción del histórico
CV3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


def servicio_vecinos(muestreo=None):
    pasos = preparacion() + [("seleccion", SelectKBest(f_classif, k=8))]
    pasos += [("muestreo", muestreo)] if muestreo is not None else []
    return ImbPipeline(pasos + [("modelo", KNeighborsClassifier(n_neighbors=25))])


filas_inst = [validar(nombre, servicio_vecinos(m), X_train, y_train, cv=CV3) for nombre, m in [
    ("Todo el histórico", None), ("CNN", CondensedNearestNeighbour(random_state=0)),
    ("ENN", EditedNearestNeighbours()), ("Tomek", TomekLinks())]]
# Muestra aleatoria del mismo tamaño que la de CNN, que también conserva a todos los que abandonan
p, c = y_train.mean(), filas_inst[1]["Conservados (%)"] / 100
aleatoria = RandomUnderSampler(sampling_strategy=p / (c - p), random_state=0)
filas_inst.append(validar("Muestra aleatoria", servicio_vecinos(aleatoria), X_train, y_train, cv=CV3))
mostrar(filas_inst)


# %% [markdown]
# ## TAREA 7
# Interpreta la tabla anterior.

# RESPUESTA (¿cuánto reduce CNN el histórico y qué cuesta? ¿compensa frente a una muestra aleatoria del mismo
# tamaño? ¿cuándo convendría reducir el histórico?):
#


# %% [markdown]
# ## 8. Modelo final y test
#
# Se comparan dos candidatos con las decisiones anteriores (variable derivada, sin datos comprados): la
# regresión logística, cuyos coeficientes permiten explicar cada llamada, y el *gradient boosting*, que trata
# los vacíos de forma nativa. El test se usa una sola vez, con el modelo elegido.


# %% 8. Candidatos
columnas = [c for c in X_train if c not in COSTE_DATOS]
candidatos = {
    "logistica": Pipeline(preparacion() + [("modelo", LogisticRegression(max_iter=1000))]),
    "boosting": Pipeline([("derivadas", FunctionTransformer(anadir_derivadas)),
                          ("modelo", HistGradientBoostingClassifier(random_state=0))]),
}
mostrar([validar(nombre, modelo, X_train[columnas], y_train) for nombre, modelo in candidatos.items()])


# %% [markdown]
# ## TAREA 8
# Elige el modelo final y ejecuta la celda siguiente una sola vez.
ELEGIDO = "logistica"  # "logistica" o "boosting"

# RESPUESTA (resumen para la dirección comercial, sin términos técnicos y en seis líneas como máximo: a quién
# llamar, resultado esperado frente a la regla actual, por qué ese modelo, qué datos dejar de comprar y qué
# riesgo vigilar):
#


# %% Evaluación en test
modelo = candidatos[ELEGIDO].fit(X_train[columnas], y_train)
mostrar([
    {"Método": f"{ELEGIDO} · test", **indicadores(y_test, modelo.predict_proba(X_test[columnas])[:, 1])},
    {"Método": "Regla actual · test", **indicadores(y_test, regla_actual(X_test))},
])

# %%
