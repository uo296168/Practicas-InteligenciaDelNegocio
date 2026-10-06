import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pareto

x = np.linspace(1, 10, 1000)
fig, ejes = plt.subplots(1, 3)
muestras = pareto.rvs(5, size=1000, random_state=42)
ejes[0].plot(x, pareto.pdf(x, 5))
ejes[0].set_title("Densidad")
ejes[1].plot(x, pareto.cdf(x, 5))
ejes[1].set_title("Distribución")
ejes[2].plot(muestras)
ejes[2].set_title("1000 muestras")
plt.tight_layout()
plt.show()
