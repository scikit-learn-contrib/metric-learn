Large Margin Nearest Neighbor (LMNN)
=====================================

LMNN learns a Mahanalobis distance metric in the kNN classification setting by semidefinite programming. The learned distance metric enforces the k-nearest neighbors to always belong to the same class while examples from different classes are separated by a large margin. This algorithm makes no assumptions about the distribution of the data.

Example Code
------------------
We use iris data here for all the examples:
::
	from sklearn.datasets import load_iris

	iris_data = load_iris()
	self.iris_points = iris_data['data']
	self.iris_labels = iris_data['target']
	np.random.seed(1234)

In this package, we have two different implementations of LMNN. Here we try both implementations in a for loop:
::
	for LMNN_cls in set((LMNN, python_LMNN)):
	      lmnn = LMNN_cls(k=k, learn_rate=1e-6)
	      lmnn.fit(self.iris_points, self.iris_labels, verbose=False)

	      csep = class_separation(lmnn.transform(), self.iris_labels)
	      self.assertLess(csep, 0.25)

References
------------------
`Distance Metric Learning for Large Margin Nearest Neighbor Classification <http://papers.nips.cc/paper/2795-distance-metric-learning-for-large-margin-nearest-neighbor-classification>`_ Kilian Q. Weinberger, John Blitzer, Lawrence K. Saul
