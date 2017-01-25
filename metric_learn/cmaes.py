# The CMA-ES algorithm takes a population of one individual as argument
# See http://www.lri.fr/~hansen/cmaes_inmatlab.html
# for more details about the rastrigin and other tests for CMA-ES

from __future__ import division, absolute_import
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance

from deap import algorithms, base, benchmarks, cma, creator, tools
from concurrent.futures import ThreadPoolExecutor 
from .base_metric import BaseMetricLearner

# This DEAP settings needs to be global because of parallelism
creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("map", ThreadPoolExecutor(max_workers=None).map)


class _BaseBuilder():
    def __init__(self):
        raise NotImplementedError('_BaseBuilder should not be instantiated')

    def _get_extra_params(self, prefixes):
        params = {}
        for pname, pvalue in self.params.items():
            if '__' not in pname: continue

            prefix, pkey = pname.split('__', 1)
            if prefix not in prefixes: continue

            params[pkey] = pvalue

        return params

    def _build_transformer(self, transformer):
        params = self._get_extra_params(('transformer', 't'))

        if isinstance(transformer, _MatrixTransformer):
            return transformer
        elif transformer == 'diagonal':
            return DiagonalMatrixTransformer(**params)
        elif transformer == 'full':
            return FullMatrixTransformer(**params)
        elif transformer == 'neuralnetwork':
            return NeuralNetworkTransformer(**params)
        elif transformer == 'kmeans':
            return KMeansTransformer(**params)
        
        raise ValueError('Invalid transformer parameter.')

    def _build_classifier(self, classifier):
        params = self._get_extra_params(('classifier', 'c'))

        if classifier == 'svc':
            return SVC(
                random_state=self.params['random_state'],
                **params,
            )
        elif classifier == 'lsvc':
            return LinearSVC(
                random_state=self.params['random_state'],
                **params,
            )
        elif classifier == 'knn':
            return KNeighborsClassifier(**params)

        raise ValueError('Invalid classifier parameter.')

class _MatrixTransformer(BaseMetricLearner):
    def __init__(self):
        raise NotImplementedError('_MatrixTransformer should not be instantiated')

    def duplicate_instance(self):
        return self.__class__(**self.params)

    def individual_size(self, input_dim):
        raise NotImplementedError('_MatrixTransformer should not be instantiated')

    def fit(self, X, y, flat_weights):
        raise NotImplementedError('_MatrixTransformer should not be instantiated')

    def transform(self, X):
        return X.dot(self.transformer().T)
        
    def transformer(self):
        return self.L

class DiagonalMatrixTransformer(_MatrixTransformer):
    def __init__(self):
        self.params = {}

    def individual_size(self, input_dim):
        return input_dim

    def fit(self, X, y, flat_weights):
        self.input_dim = X.shape[1]
        if self.input_dim != len(flat_weights):
            raise Error('Invalid size of input_dim')

        self.L = np.diag(flat_weights)
        return self

class FullMatrixTransformer(_MatrixTransformer):
    def __init__(self, n_components=None):
        self.params = {
            'n_components': n_components
        }

    def individual_size(self, input_dim):
        if self.params['n_components'] is None:
            return input_dim**2

        return input_dim*self.params['n_components']

    def fit(self, X, y, flat_weights):
        input_dim = X.shape[1]
        if self.individual_size(input_dim) != len(flat_weights):
            raise Error('input_dim and flat_weights sizes do not match')

        self.L = np.reshape(flat_weights, (len(flat_weights)//input_dim, input_dim))
        return self

class NeuralNetworkTransformer(_MatrixTransformer):
    def __init__(self, layers=None, activation='relu', use_biases=False):
        self.params = {
            'layers': layers,
            'activation': activation,
            'use_biases': use_biases,
        }

    def _build_activation(self):
        activation = self.params['activation']

        if activation is None:
            return lambda X: X # identity
        elif activation == 'relu':
            return lambda X: np.maximum(X, 0) # ReLU
        elif activation == 'tanh':
            return np.tanh
        else:
            raise ValueError('Invalid activation paramater value')

    def individual_size(self, input_dim):
        last_layer = input_dim

        size = 0
        for layer in self.params['layers'] or (input_dim,):
            size += last_layer*layer

            if self.params['use_biases']:
                size += layer

            last_layer = layer

        return size

    def fit(self, X, y, flat_weights):
        input_dim = X.shape[1]

        flat_weights = np.array(flat_weights)
        flat_weights_len = len(flat_weights)
        if flat_weights_len != self.individual_size(input_dim):
            raise Error('Invalid size of the flat_weights')

        weights = []

        last_layer = input_dim
        offset = 0
        for layer in self.params['layers'] or (input_dim,):
            W = flat_weights[offset:offset+last_layer*layer].reshape((last_layer, layer))
            offset += last_layer*layer
            
            if self.params['use_biases']:
                b = flat_weights[offset:offset+layer]
                offset += layer
            else:
                b = np.zeros((layer))

            assert(offset <= flat_weights_len)
            weights.append( (W, b) )
            last_layer = layer

        self._parsed_weights = weights
        self._activation = self._build_activation()
        return self

    def transform(self, X):
        for i, (W, b) in enumerate(self._parsed_weights):
            X = np.add(np.matmul(X, W), b)

            if i+1 < len(self._parsed_weights):
                X = self._activation(X)

        return X
        
    def transformer(self):
        return self._parsed_weights

class KMeansTransformer(_MatrixTransformer, _BaseBuilder):
    def __init__(self, transformer='full', n_clusters='classes', function='distance', n_init=1, random_state=None, **kwargs):
        self._transformer = None
        self.params = {
            **kwargs,
            'transformer': transformer,
            'n_clusters': n_clusters,
            'function': function,
            'n_init': n_init,
            'random_state': random_state,
        }

    def individual_size(self, input_dim):
        if self._transformer is None:
            self._transformer = self._build_transformer(self.params['transformer'])

        return self._transformer.individual_size(input_dim)

    def fit(self, X, y, flat_weights):
        self._transformer = self._build_transformer(self.params['transformer'])
        self._transformer.fit(X, y, flat_weights)

        if self.params['n_clusters'] == 'classes':
            n_clusters = np.unique(y).size
        elif self.params['n_clusters'] == 'same':
            n_clusters = X.shape[1]
        else:
            n_clusters = int(self.params['n_clusters']) # may raise an exception

        self.kmeans = KMeans(
            n_clusters = n_clusters,
            random_state = self.params['random_state'],
            n_init = self.params['n_init'],
        )
        self.kmeans.fit(X)
        self.centers = self.kmeans.cluster_centers_

        return self

    def transform(self, X):
        Xt = self._transformer.transform(X)

        if self.params['function'] == 'distance':
            return distance.cdist(Xt, self.centers)
        elif self.params['function'] == 'product':
            return np.dot(Xt, self.centers.T)

        raise ValueError('Invalid function param.')
        
    def transformer(self):
        return self._transformer

class CMAES(BaseMetricLearner, _BaseBuilder):
    '''
    CMAES
    '''
    def __init__(self, classifier='knn', transformer='full', n_gen=25, class_separation=False,
                 train_subset_size=1.0, split_size=0.33,
                 random_state=None, verbose=False, **kwargs):
        """Initialize the learner.

        Parameters
        ----------
        classifier : ('knn', 'svc', 'lsvc', classifier object)
            classifier is used in fitness scoring
        transformer : ('full', 'diagonal', _MatrixTransformer object)
            transformer defines transforming function to learn
        n_gen : int, optional
            number of generations of evolution algorithm
        class_separation : bool, optional
            add class separation to fitness
        train_subset_size : float (0,1], optional
            size of data to use for individual evaluation
        split_size : float (0,1], optional
            size of the data to leave for test error
        verbose : bool, optional
            if True, prints information while learning
        """
        self.params = {
            **kwargs,
            'classifier': classifier,
            'transformer': transformer,
            'n_gen': n_gen,
            'class_separation': class_separation,
            'train_subset_size': train_subset_size,
            'split_size': split_size,
            'random_state': random_state,
            'verbose': verbose,
        }

    def transform(self, X):
        return self._transformer.transform(X)
        
    def transformer(self):
        return self._transformer

    def fit_transform(self, X, Y):
        self.fit(X,Y)
        return self.transform(X)

    def evaluation_builder(self, X, y):
        def class_separation(X, labels):
            unique_labels, label_inds = np.unique(labels, return_inverse=True)
            ratio = 0
            for li in range(len(unique_labels)):
                Xc = X[label_inds==li]
                Xnc = X[label_inds!=li]
                ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc,Xnc).mean()
            return ratio / len(unique_labels)

        def evaluate(individual):
            subset = self.params['train_subset_size']
            train_mask = np.random.choice([True, False], X.shape[0], p=[subset, 1-subset])
            X_train, X_test, y_train, y_test = train_test_split(X[train_mask], y[train_mask], test_size=self.params['split_size'], random_state=self.params['random_state'])

            transformer = self._transformer.duplicate_instance().fit(X_train, y_train, individual)
            X_train_trans = transformer.transform(X_train)
            X_test_trans = transformer.transform(X_test)

            classifier = self._build_classifier(self.params['classifier'])
            classifier.fit(X_train_trans, y_train)
            score = classifier.score(X_test_trans, y_test)

            if self.params['class_separation']:
                return (score, class_separation(X_test_trans, y_test),)
            else:
                return (score, 0,)

            # return [score, self.params['class_separation']*separation_score]
            # return [score - mean_squared_error(individual, np.ones(self._input_dim))]
            # return [score - np.sum(np.absolute(individual))]
        return evaluate

    def fit(self, X, Y):
        '''
         X: (n, d) array-like of samples
         Y: (n,) array-like of class labels
        '''
        np.random.seed(self.params['random_state'])
        self._transformer = self._build_transformer(self.params['transformer'])

        sizeOfIndividual = self._transformer.individual_size(X.shape[1])
        
        strategy = cma.Strategy(centroid=[0.0]*sizeOfIndividual, sigma=1.0)
        toolbox.register("evaluate", self.evaluation_builder(X, Y))
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        self.hof = tools.HallOfFame(1)
        
        if self.params['verbose']:
            stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
            stats_fit.register("avg", np.mean, axis=0)
            stats_fit.register("std", np.std, axis=0)
            stats_fit.register("min", np.min, axis=0)
            stats_fit.register("max", np.max, axis=0)
            
            stats_size = tools.Statistics()
            stats_size.register("x", lambda x: x)

            stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
            stats = stats_fit
        else:
            stats=None

        pop, logbook = algorithms.eaGenerateUpdate(
            toolbox,
            ngen=self.params['n_gen'],
            stats=stats,
            halloffame=self.hof,
            verbose=self.params['verbose']
        )
        
        self._transformer.fit(X, Y, self.hof[0])
        return self
