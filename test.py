from metric_learn import RCA_Supervised
import numpy as np

X = np.random.rand(5, 2)
y = [1, 1, -1, 2, 2]

rca = RCA_Supervised(num_chunks=2)
rca.fit(X, y)
