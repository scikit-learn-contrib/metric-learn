import numpy as np
from sklearn.datasets import load_iris
from metric_learn import SDML_Supervised
import matplotlib.pyplot as plt

dataset = load_iris()
X, y = dataset.data, dataset.target
sdml = SDML_Supervised(num_constraints=200)
sdml.fit(X, y, random_state = np.random.RandomState(1234))
embeddings = sdml.transform(X)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

plt.figure()
plt.scatter(embeddings[:, 0], embeddings[:, 1], c=y)
plt.show()