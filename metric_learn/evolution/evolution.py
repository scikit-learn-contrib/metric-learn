# The CMA-ES algorithm takes a population of one individual as argument
# See http://www.lri.fr/~hansen/cmaes_inmatlab.html
# for more details about the rastrigin and other tests for CMA-ES

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class MetricEvolution(BaseEstimator, TransformerMixin):
    def __init__(self, strategy, fitness, transformer_func,
                 random_state=None, verbose=False):
        """Initialize the learner.

        Parameters
        ----------
        fitness : ('knn', 'svc', 'lsvc', fitness object)
            fitness is used in fitness scoring
        transformer_func : ('full', 'diagonal', MatrixTransformer object)
            transformer_func shape defines transforming function to learn
        verbose : bool, optional
            if True, prints information while learning
        """
        self._strategy = strategy
        self._fitness = fitness
        self._transformer = transformer_func
        self.random_state = random_state
        self.verbose = verbose

        np.random.seed(random_state)

    def transform(self, X):
        """Applies the metric transformation.

        Parameters
        ----------
        X : (n x d) matrix
            Data to transform.

        Returns
        -------
        transformed : (n x d) matrix
            Input data transformed to the metric space by :math:`XL^{\\top}`
        """
        X = check_array(X, accept_sparse=True)
        return self._transformer.transform(X)

    def transformer(self):
        """Computes the transformation matrix from the Mahalanobis matrix.

        Returns
        -------
        L : (d x d) matrix
        """
        return self._transformer.transformer()

    def transformer_class(self):
        return self._transformer

    def metric(self):
        """Computes the Mahalanobis matrix from the transformation matrix.

        .. math:: M = L^{\\top} L

        Returns
        -------
        M : (d x d) matrix
        """
        return self._transformer.metric()

    def fit(self, X, y):
        '''
         X: (n, d) array-like of samples
         Y: (n,) array-like of class labels
        '''

        # Inject parameters into all fitness functions
        for f in self._fitness:
            f.inject_params(
                random_state=self.random_state,
            )

        # Inject parameters into Strategy
        self._strategy.inject_params(
            n_dim=self._transformer.individual_size(X.shape[1]),
            fitness=self._fitness,
            transformer=self._transformer,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Evolve best transformer
        self._strategy.fit(X, y)

        # Fit transformer with the best individual
        self._transformer.fit(X, y, self._strategy.best_individual())
        return self
