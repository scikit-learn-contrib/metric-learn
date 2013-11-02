# metric_learn

Metric Learning algorithms in Python.

**Algorithms**

 * Large Margin Nearest Neighbor (LMNN)
 * Information Theoretic Metric Learning (ITML)
 * Sparse Determinant Metric Learning (SDML)
 * Least Squares Metric Learning (LSML)

For usage examples, see `test.py` and `demo.py`.

**Dependencies**

 * Python 2.6+
 * numpy, scipy, scikit-learn
 * (for the demo only: matplotlib)

**Notes**

If a recent version of the Shogun Python modular (`modshogun`) library is available,
the LMNN implementation will use the fast C++ version from there.
The two implementations differ slightly, and the C++ version is more complete.
