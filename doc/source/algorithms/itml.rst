Information Theoretic Metric Learning (ITML)
=====================================
ITML minimizes the differential relative entropy between two multivariate Gaussians under constraints on the distance function, which can be formulated into a particular Bregman optimization problem by minimizing the LogDet divergence subject to the linear constraints. This algorithm can handle wide variety of constraints and can optionally incorporate a prior on the distance function. Unlike some existing method, ITML is fast and scalable since no eigenvalue computation or semi-definite programming are required. 

Example Code
------------------
After loading the iris data, we apply ITML:
::
	from metric_learn import ITML

	num_constraints = 200
	n = self.iris_points.shape[0]
	C = ITML.prepare_constraints(self.iris_labels, n, num_constraints)
	itml = ITML().fit(self.iris_points, C, verbose=False)
	itml.transform()

References
------------------
`Information-theoretic Metric Learning <http://machinelearning.wustl.edu/mlpapers/paper_files/icml2007_DavisKJSD07.pdf>`_ Jason V. Davis, et al.