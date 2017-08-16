# The CMA-ES algorithm takes a population of one individual as argument
# See http://www.lri.fr/~hansen/cmaes_inmatlab.html
# for more details about the rastrigin and other tests for CMA-ES

from __future__ import absolute_import, division

import numpy as np
from numpy.linalg import cholesky

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array

from . import fitness as fit
from . import strategy as st
from . import transformer as tr


class BaseBuilder():
    def __init__(self):
        raise NotImplementedError('BaseBuilder should not be instantiated')

    # def _get_extra_params(self, prefixes):
    #     params = {}
    #     for pname, pvalue in self.params.items():
    #         if '__' not in pname: continue

    #         prefix, pkey = pname.split('__', 1)
    #         if prefix not in prefixes: continue

    #         params[pkey] = pvalue

    #     return params

    def build_transformer(self, transformer_shape=None, num_dims=None,
                          params={}):
        if transformer_shape is None:
            transformer_shape = self.transformer_shape
        # if params is None:
            # params = self._get_extra_params(('transformer_shape', 't'))

        if isinstance(transformer_shape, tr.MatrixTransformer):
            return transformer_shape
        elif transformer_shape == 'diagonal':
            return tr.DiagonalMatrixTransformer(**params)
        elif transformer_shape == 'full':
            return tr.FullMatrixTransformer(
                num_dims=num_dims, **params)
        elif transformer_shape == 'triangular':
            return tr.TriangularMatrixTransformer(**params)
        elif transformer_shape == 'neuralnetwork':
            return tr.NeuralNetworkTransformer(**params)
        elif transformer_shape == 'kmeans':
            return tr.KMeansTransformer(**params)

        raise ValueError(
            'Invalid `transformer_shape` parameter value: `{}`'
            .format(transformer_shape))


class BaseMetricLearner(BaseEstimator, TransformerMixin):
    def __init__(self):
        raise NotImplementedError('BaseMetricLearner should not be instantiated')

    def metric(self):
        """Computes the Mahalanobis matrix from the transformation matrix.

        .. math:: M = L^{\\top} L

        Returns
        -------
        M : (d x d) matrix
        """
        L = self.transformer()
        return L.T.dot(L)

    def transformer(self):
        """Computes the transformation matrix from the Mahalanobis matrix.

        L = cholesky(M).T

        Returns
        -------
        L : upper triangular (d x d) matrix
        """
        return cholesky(self.metric()).T

    def transform(self, X=None):
        """Applies the metric transformation.

        Parameters
        ----------
        X : (n x d) matrix, optional
            Data to transform. If not supplied, the training data will be used.

        Returns
        -------
        transformed : (n x d) matrix
            Input data transformed to the metric space by :math:`XL^{\\top}`
        """
        if X is None:
            X = self.X_
        else:
            X = check_array(X, accept_sparse=True)
        L = self.transformer()
        return X.dot(L.T)


class MetricEvolution(BaseMetricLearner, BaseBuilder):
    def __init__(self, strategy='cmaes', fitnesses='knn', transformer_shape='full',
                 num_dims=None, random_state=None, verbose=False):
        """Initialize the learner.

        Parameters
        ----------
        fitnesses : ('knn', 'svc', 'lsvc', fitnesses object)
            fitnesses is used in fitness scoring
        transformer_shape : ('full', 'diagonal', MatrixTransformer object)
            transformer shape defines transforming function to learn
        num_dims : int, optional
            Dimensionality of reduced space (defaults to dimension of X)
        verbose : bool, optional
            if True, prints information while learning
        """
        if (num_dims is not None) and (transformer_shape != 'full'):
            raise Exception(
                '`num_dims` can be only set for `transformer_shape`=="full"')

        self.strategy = strategy
        self.fitnesses = fitnesses
        self.transformer_shape = transformer_shape
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
            return fit.WeightedPurityFitness(**params)

        if fit.WeightedFMeasureFitness.available(fitness):
            return fit.WeightedFMeasureFitness(**params)

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
            return st.CMAES(**strategy_params)
        elif strategy == 'de':
            return st.DifferentialEvolution(**strategy_params)
        elif strategy == 'jde':
            return st.SelfAdaptingDifferentialEvolution(
                **strategy_params)
        elif strategy == 'dde':
            return st.DynamicDifferentialEvolution(**strategy_params)

        raise ValueError('Invalid `strategy` parameter value.')

    def transform(self, X):
        return self._transformer.transform(X)

    def transformer_class(self):
        return self._transformer

    def transformer(self):
        return self._transformer.transformer()

    def metric(self):
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
