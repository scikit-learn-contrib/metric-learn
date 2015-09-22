Large Margin Nearest Neighbor (LMNN)
=====================================

todo: Brief desciption or a link.

Example Code
=====================================
Using two different implementations:

::

	for LMNN_cls in set((LMNN, python_LMNN)):
	      lmnn = LMNN_cls(k=k, learn_rate=1e-6)
	      lmnn.fit(self.iris_points, self.iris_labels, verbose=False)

	      csep = class_separation(lmnn.transform(), self.iris_labels)
	      self.assertLess(csep, 0.25)

