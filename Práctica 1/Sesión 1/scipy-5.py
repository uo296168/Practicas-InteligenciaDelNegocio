import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import minkowski

rejilla = np.meshgrid(
    np.linspace(-1.1, 1.1, 512),
    np.linspace(-1.1, 1.1, 512),
    indexing="ij",
)
X, Y = rejilla

def en_bola(x, y, p):
    return minkowski([x, y], [0, 0], p) <= 1
def bola(p):
    return np.vectorize(en_bola)(X, Y, p)

plt.imshow(
    bola(3),
    extent=[-1.1, 1.1, -1.1, 1.1],
    origin="lower",
    cmap="Blues",
)
plt.gca().set_aspect("equal")
plt.axis("off")
plt.show()
