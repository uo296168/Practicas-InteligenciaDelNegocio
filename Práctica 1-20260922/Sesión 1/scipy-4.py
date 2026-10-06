import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss

derecha = np.array([
    113, 105, 130, 101, 138, 118, 87, 116, 75, 96,
    122, 103, 116, 107, 118, 103, 111, 104, 111, 89,
    78, 100, 89, 85, 88,
])
izquierda = np.array([
    137, 105, 133, 108, 115, 170, 103, 145, 78, 107,
    84, 148, 147, 87, 166, 146, 123, 135, 112, 93,
    76, 116, 78, 101, 123,
])

diferencias = izquierda - derecha
media, desv = ss.norm.fit(diferencias)
x = np.linspace(diferencias.min(), diferencias.max(), 1000)
normal = ss.norm.pdf(x, media, desv)
kde = ss.gaussian_kde(diferencias)

plt.hist(
    diferencias,
    density=True,
    alpha=0.45,
    edgecolor="white",
    label="Datos",
)
plt.plot(x, normal, label="Normal")
plt.plot(x, kde(x), label="KDE")
plt.legend()
plt.tight_layout()
plt.show()
