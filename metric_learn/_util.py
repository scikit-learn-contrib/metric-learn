import numpy as np


# hack around lack of axis kwarg in older numpy versions
try:
    np.linalg.norm([[4]], axis=1)
except TypeError:
    def vector_norm(X):
        return np.apply_along_axis(np.linalg.norm, 1, X)
else:
    def vector_norm(X):
        return np.linalg.norm(X, axis=1)
