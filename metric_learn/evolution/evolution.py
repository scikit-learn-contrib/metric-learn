"""
MetricEvolution is a modular interface for evolving a transformation function

It joins together three important parts:
    - transformation function (typically Mahalanobis metric matrix)
    - fitness function(s) describing the quality of the transformation
    - evolution strategy used for finding the best transformation
      according to a given fitness function(s)

Because an evolutionary strategy is used to find
the best transformation function,
the fitness function does not have to be differentiable
and the transformation function can be any function
parametrized by a vector of real values.
"""

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class MetricEvolution(BaseEstimator, TransformerMixin):
    """
    Modular interface for evolving a transformation function
    """
    def __init__(self, strategy, fitness_list, transformer_func,
                 random_state=None, verbose=False):
        """Initialize the evolutionary learner.

        Parameters
        ----------
        strategy : BaseEvolutionStrategy object
            Evolution strategy used for the optimization
        fitness_list : list of BaseFitness objects
            Fitnesses are used during the evolution to evaluate the metric
        transformer_func : BaseTransformer object
            Defines a transforming function to be learnt
        random_state : numpy.random.RandomState, optional
            If provided, controls random number generation.
        verbose : bool, optional
            if True, prints information while learning
        """
        self._strategy = strategy
        self._fitness = fitness_list
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
        transformed : (n x D) matrix
            Input data transformed to the metric space by transformer function`
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
        """Getter for transformer object

        Returns
        -------
        transformer object inheriting from BaseTransformer object
        """
        return self._transformer

    def metric(self):
        """Computes the Mahalanobis matrix from the transformation matrix if available.

        .. math:: M = L^{\\top} L

        Returns
        -------
        M : (d x d) matrix
        """
        return self._transformer.metric()

    def fit(self, X, y):
        '''Fit the model.

        Parameters
        ----------
        X : (n, d) array-like
            Input data.

        y : (n,) array-like
            Class labels, one per point of data.
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

        # transformer functions using strategy by optimising _fitnesses
        self._strategy.fit(X, y)

        # Fit (fill) transformer with the weights from the best individual
        self._transformer.fit(X, y, self._strategy.best_individual())
        return self
