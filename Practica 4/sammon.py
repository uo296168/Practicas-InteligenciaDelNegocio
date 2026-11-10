"""Función de Sammon suministrada en la clase 1 de octubre de 2026.
Extraída de sesion1_distancias.py. Se requiere indicar un mapa inicial Y0.
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform

def distancias(X):
    return squareform(pdist(X))

def sammon(D, p=2, Y0=None, n_iter=150, factor=0.35, tol=1e-6):
    """Cartografía de Sammon. Devuelve el mapa y el valor de E en cada iteración."""
    n = D.shape[0]
    I = np.eye(n)
    Dm = D + I                                     # unos en la diagonal: evita divisiones por cero
    c = D.sum() / 2

    def esfuerzo(Y):
        d = distancias(Y) + I
        return ((Dm - d) ** 2 / Dm).sum() / (2 * c), d

    if Y0 is None:
        raise ValueError("Indica Y0, por ejemplo el mapa de PCA")
    Y = Y0.copy()
    E, d = esfuerzo(Y)
    historia = [E]
    for _ in range(n_iter):
        delta = Dm - d
        q = delta / (Dm * d)
        np.fill_diagonal(q, 0)
        gradiente = -(2 / c) * (q.sum(axis=1)[:, None] * Y - q @ Y)
        r = (1 + delta / d) / (Dm * d * d)
        np.fill_diagonal(r, 0)
        hessiano = np.empty_like(Y)                # solo la diagonal
        for a in range(p):
            dif2 = (Y[:, a][:, None] - Y[:, a][None, :]) ** 2
            hessiano[:, a] = -(2 / c) * (q.sum(axis=1) - (r * dif2).sum(axis=1))
        paso = factor * gradiente / np.abs(hessiano)
        for _ in range(12):                        # se reduce el paso si E no disminuye
            E_nuevo, d_nuevo = esfuerzo(Y - paso)
            if E_nuevo < E:
                break
            paso *= 0.5
        Y, d = Y - paso, d_nuevo
        historia.append(E_nuevo)
        convergido = abs(E - E_nuevo) < tol * E
        E = E_nuevo
        if convergido:
            break
    return Y, np.array(historia)
