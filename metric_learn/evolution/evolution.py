# The CMA-ES algorithm takes a population of one individual as argument
# See http://www.lri.fr/~hansen/cmaes_inmatlab.html
# for more details about the rastrigin and other tests for CMA-ES

from __future__ import absolute_import, division

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from . import fitness as fit
from . import strategy as st
from . import transformer as tr


class MetricEvolution(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='cmaes', fitnesses='knn', transformer='full',
                 num_dims=None, random_state=None, verbose=False):
        """Initialize the learner.

        Parameters
        ----------
        fitnesses : ('knn', 'svc', 'lsvc', fitnesses object)
            fitnesses is used in fitness scoring
        transformer : ('full', 'diagonal', MatrixTransformer object)
            transformer shape defines transforming function to learn
        num_dims : int, optional
            Dimensionality of reduced space (defaults to dimension of X)
        verbose : bool, optional
            if True, prints information while learning
        """
        if (num_dims is not None) and (transformer != 'full'):
            raise Exception(
                '`num_dims` can be only set for `transformer`=="full"')

        self.strategy = strategy
        self.fitnesses = fitnesses
        self.transformer = transformer
        self.num_dims = num_dims
        self.random_state = random_state
        self.verbose = verbose

        np.random.seed(random_state)

    def build_fitnesses(self, fitnesses):
        # make it an array if it is not already one
        if not isinstance(fitnesses, (list, tuple)):
            fitnesses = [fitnesses]

        return list(map(self.build_fitness, fitnesses))

    def build_fitness(self, fitness, params={}):
        # fitness can be a tuple of fitness and its params
        if isinstance(fitness, (list, tuple)):
            fitness, params = fitness
        # else:
            # params = self._get_extra_params(('fitness', 'f'))
        if fit.ClassifierFitness.available(fitness):
            return fit.ClassifierFitness(fitness, **params)

        if fit.RandomFitness.available(fitness):
            return fit.RandomFitness()

        if fit.ClassSeparationFitness.available(fitness):
            return fit.ClassSeparationFitness()

        if fit.WeightedPurityFitness.available(fitness):
            return fit.WeightedPurityFitness(random_state=self.random_state, **params)

        if fit.WeightedFMeasureFitness.available(fitness):
            return fit.WeightedFMeasureFitness(random_state=self.random_state, **params)

        # TODO unify error messages
        raise ValueError('Invalid value of fitness: `{}`'.format(fitness))

    def build_strategy(self, strategy, strategy_params, fitnesses, n_dim,
                       transformer, random_state, verbose):
        strategy_params = {
            'fitnesses': fitnesses,
            'n_dim': n_dim,
            'transformer': transformer,
            'random_state': random_state,
            'verbose': verbose,
        }

        if isinstance(strategy, st.BaseEvolutionStrategy):
            return strategy
        elif strategy == 'cmaes':
            return st.CMAESEvolution(**strategy_params)
        elif strategy == 'de':
            return st.DifferentialEvolution(**strategy_params)
        elif strategy == 'jde':
            return st.SelfAdaptingDifferentialEvolution(
                **strategy_params)
        elif strategy == 'dde':
            return st.DynamicDifferentialEvolution(**strategy_params)

        raise ValueError('Invalid `strategy` parameter value.')

    def build_transformer(self, transformer=None, num_dims=None,
                          params={}):
        if transformer is None:
            transformer = self.transformer
        # if params is None:
            # params = self._get_extra_params(('transformer', 't'))

        if isinstance(transformer, tr.MatrixTransformer):
            return transformer
        elif transformer == 'diagonal':
            return tr.DiagonalMatrixTransformer(**params)
        elif transformer == 'full':
            return tr.FullMatrixTransformer(
                num_dims=num_dims, **params)
        elif transformer == 'triangular':
            return tr.TriangularMatrixTransformer(**params)
        elif transformer == 'neuralnetwork':
            return tr.NeuralNetworkTransformer(**params)
        elif transformer == 'kmeans':
            return tr.KMeansTransformer(**params)

        raise ValueError(
            'Invalid `transformer` parameter value: `{}`'
            .format(transformer))

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

    def transformer_class(self):
        return self._transformer

    def transformer(self):
        """Computes the transformation matrix from the Mahalanobis matrix.

        Returns
        -------
        L : (d x d) matrix
        """
        return self._transformer.transformer()

    def metric(self):
        """Computes the Mahalanobis matrix from the transformation matrix.

        .. math:: M = L^{\\top} L

        Returns
        -------
        M : (d x d) matrix
        """
        return self._transformer.metric()

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y):
        '''
         X: (n, d) array-like of samples
         Y: (n,) array-like of class labels
        '''
        # Initialize Transformer builder and default transformer
        self._transformer = self.build_transformer(num_dims=self.num_dims)

        # Build strategy and fitnesses with correct params
        self._strategy = self.build_strategy(
            strategy=self.strategy,
            strategy_params={},  # self._get_extra_params(('strategy', 's')),
            fitnesses=self.build_fitnesses(self.fitnesses),
            n_dim=self._transformer.individual_size(X.shape[1]),
            transformer=self._transformer,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Evolve best transformer
        self._strategy.fit(X, y)

        # Fit transformer with the best individual
        self._transformer.fit(X, y, self._strategy.best_individual())
        return self
