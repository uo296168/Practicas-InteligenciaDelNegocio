# -*- coding: utf-8 -*-
"""Práctica 2: indicadores de la campaña y comparación de alternativas por validación cruzada."""
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.parallel import Parallel, delayed

# Economía de la campaña
VALOR_CLIENTE = 600  # margen que se pierde cuando un cliente se marcha (€)
COSTE_LLAMADA = 40   # coste de una llamada con oferta (€)
EXITO_OFERTA = 0.30  # fracción de quienes iban a marcharse que se quedan tras la llamada
CAPACIDAD = 0.15     # fracción de la cartera que se puede llamar
GANANCIA = EXITO_OFERTA * VALOR_CLIENTE - COSTE_LLAMADA  # 140 € por llamada útil

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
COLUMNAS = ["Precisión (%)", "Exhaustividad (%)", "Beneficio (€/1.000)", "± (€)", "Mejora", "Ajuste (s)"]


def indicadores(y, riesgo):
    """Llama a la fracción CAPACIDAD de clientes con mayor riesgo y mide el resultado de la campaña."""
    y = np.asarray(y)
    n_llamadas = int(round(CAPACIDAD * len(y)))
    llamados = np.argsort(-np.asarray(riesgo, dtype=float), kind="stable")[:n_llamadas]
    utiles = y[llamados].sum()  # llamadas a clientes que iban a marcharse
    beneficio = utiles * GANANCIA - (n_llamadas - utiles) * COSTE_LLAMADA
    return {
        "Precisión (%)": 100 * utiles / n_llamadas,
        "Exhaustividad (%)": 100 * utiles / y.sum(),
        "Beneficio (€/1.000)": 1000 * beneficio / len(y),
    }


def beneficio(y, riesgo):
    return indicadores(y, riesgo)["Beneficio (€/1.000)"]


criterio_beneficio = make_scorer(beneficio, response_method="predict_proba")  # para RFECV


def _pliegue(modelo, X, y, entrena, valida, solo_completos):
    X_e, y_e, X_v, y_v = X.iloc[entrena], y.iloc[entrena], X.iloc[valida], y.iloc[valida]
    puntuables = np.ones(len(X_v), dtype=bool)
    if solo_completos:  # eliminación: se entrena y se puntúa solo a clientes sin datos ausentes
        completos = X_e.notna().all(axis=1)
        X_e, y_e = X_e[completos], y_e[completos]
        puntuables = X_v.notna().all(axis=1).to_numpy()
    t0 = time.perf_counter()
    ajustado = clone(modelo).fit(X_e, y_e)
    fila = {"Ajuste (s)": time.perf_counter() - t0}
    riesgo = np.full(len(X_v), -np.inf)  # un cliente sin puntuación nunca recibe llamada
    riesgo[puntuables] = ajustado.predict_proba(X_v[puntuables])[:, 1]
    fila.update(indicadores(y_v, riesgo))
    if solo_completos:
        fila["Sin puntuar (%)"] = 100 * (1 - puntuables.mean())
    muestreo = getattr(ajustado, "named_steps", {}).get("muestreo")
    if muestreo is not None:
        fila["Conservados (%)"] = 100 * len(muestreo.sample_indices_) / len(X_e)
    return fila


def validar(nombre, modelo, X, y, cv=CV, solo_completos=False):
    """Validación cruzada: en cada pliegue el pipeline se ajusta solo con la parte de entrenamiento."""
    filas = Parallel(n_jobs=-1)(delayed(_pliegue)(modelo, X, y, e, v, solo_completos) for e, v in cv.split(X, y))
    pliegues = pd.DataFrame(filas)
    return {"Método": nombre, **pliegues.mean().to_dict(), "_beneficios": pliegues["Beneficio (€/1.000)"].tolist()}


def evaluar_regla(nombre, regla, X, y, cv=CV):
    """Valida una regla fija (sin aprendizaje) en los mismos pliegues que los modelos."""
    pliegues = pd.DataFrame([indicadores(y.iloc[v], regla(X.iloc[v])) for _, v in cv.split(X, y)])
    return {"Método": nombre, **pliegues.mean().to_dict(), "_beneficios": pliegues["Beneficio (€/1.000)"].tolist()}


def mostrar(filas):
    """Imprime los resultados. ±: desviación típica del beneficio entre pliegues.
    Mejora: pliegues en que el método supera en beneficio a la primera fila."""
    t = pd.DataFrame(filas).set_index("Método")
    if "_beneficios" in t:
        ref = t["_beneficios"].iloc[0]
        t["± (€)"] = [np.std(b, ddof=1) if isinstance(b, list) else np.nan for b in t["_beneficios"]]
        t["Mejora"] = ["ref."] + [f"{sum(a > r for a, r in zip(b, ref))}/{len(ref)}" if isinstance(b, list) else "—"
                                  for b in t["_beneficios"].iloc[1:]]
        t = t.drop(columns="_beneficios")
    t = t[[c for c in COLUMNAS if c in t] + [c for c in t if c not in COLUMNAS]]
    print(t.round(1).round({"Beneficio (€/1.000)": 0, "± (€)": 0, "Ajuste (s)": 2}).to_string(), "\n")


def error_saldo(imputador, X, ids, recuperados):
    """Error mediano (€) al imputar saldo_hace_3m en los clientes de X cuyo saldo real se recuperó."""
    prep = Pipeline([("escalado", StandardScaler()), ("imputacion", clone(imputador))]).fit(X)
    escalado = prep.named_steps["escalado"]
    Z = prep.transform(X)[escalado.get_feature_names_out()]  # sin las columnas indicadoras
    euros = pd.DataFrame(escalado.inverse_transform(Z), columns=Z.columns, index=X.index)
    reales = recuperados.set_index("id_cliente")["saldo_hace_3m"]
    con_real = ids.isin(reales.index).to_numpy()
    error = np.abs(euros["saldo_hace_3m"].to_numpy()[con_real] - reales.loc[ids[con_real]].to_numpy())
    return {"Error saldo (€)": np.median(error)}
