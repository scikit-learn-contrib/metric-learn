# The CMA-ES algorithm takes a population of one individual as argument
# See http://www.lri.fr/~hansen/cmaes_inmatlab.html
# for more details about the rastrigin and other tests for CMA-ES

from __future__ import division, absolute_import
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import pairwise_distances

from deap import algorithms, base, benchmarks, cma, creator, tools
from concurrent.futures import ThreadPoolExecutor 
from .base_metric import BaseMetricLearner

# This DEAP settings needs to be global because of parallelism
creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("map", ThreadPoolExecutor(max_workers=None).map)


class _MatrixTransformer(BaseMetricLearner):
    def __init__(self):
        raise NotImplementedError('_MatrixTransformer should not be instantiated')

    def duplicate_instance(self):
        return self.__class__(**self.params)

    def individual_size(self, input_dim):
        raise NotImplementedError('_MatrixTransformer should not be instantiated')

    def fit(self, input_dim, flat_weights):
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

    def fit(self, input_dim, flat_weights):
        self.input_dim = input_dim
        if input_dim != len(flat_weights):
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

    def fit(self, input_dim, flat_weights):

        if self.individual_size(input_dim) != len(flat_weights):
            raise Error('input_dim and flat_weights sizes do not match')

        self.L = np.reshape(flat_weights, (len(flat_weights)//input_dim, input_dim))
        return self

class NeuralNetworkTransformer(BaseMetricLearner):
    def __init__(self, layers, activation='relu', use_biases=False):
        self.params = {
            'layers': layers,
            'activation': activation,
            'use_biases': use_biases,
        }

    def duplicate_instance(self):
        return self.__class__(**self.params)

    def individual_size(self, input_dim):
        last_layer = input_dim

        size = 0
        for layer in self.params['layers']:
            size += last_layer*layer

            if self.params['use_biases']:
                size += layer

            last_layer = layer

        return size

    def fit(self, input_dim, flat_weights):
        flat_weights = np.array(flat_weights)
        flat_weights_len = len(flat_weights)
        if flat_weights_len != self.individual_size(input_dim):
            raise Error('Invalid size of the flat_weights')

        weights = []

        last_layer = input_dim
        offset = 0
        for layer in self.params['layers']:
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
        return self

    def transform(self, X):
        for i, (W, b) in enumerate(self._parsed_weights):
            X = np.add(np.matmul(X, W), b)

            if i+1 < len(self._parsed_weights):
                if self.params['activation']=='relu':
                    X = np.maximum(X, 0) # ReLU
                elif self.params['activation']=='tanh':
                    X = np.tanh(X) # tanh
                else:
                    pass

        return X
        
    def transformer(self):
        return self._parsed_weights

class CMAES(BaseMetricLearner):
    '''
    CMAES
    '''
    def __init__(self, transformer, n_gen=25, n_neighbors=1, classifier='knn', class_separation=0,
                 knn_weights='uniform', train_subset_size=1.0, split_size=0.33,
                 n_jobs=-1, random_state=None, verbose=False):
        """Initialize the learner.

        Parameters
        ----------
        transformer : _MatrixTransformer object
            transformer defines transforming function to learn
        n_gen : int, optional
            number of generations of evolution algorithm
        n_neighbors : int, optional
            number of neighbors to use for k_neighbors queries
        class_separation : bool, optional
            add class separation to fitness
        knn_weights : [uniform, distance], optional
            weight function used in prediction (for Scikit's KNeighborsClassifier)
        train_subset_size : float (0,1], optional
            size of data to use for individual evaluation
        split_size : float (0,1], optional
            size of the data to leave for test error
        verbose : bool, optional
            if True, prints information while learning
        """
        self.params = {
            'transformer': transformer,
            'n_gen': n_gen,
            'n_neighbors': n_neighbors,
            'classifier': classifier,
            'class_separation': class_separation,
            'knn_weights': knn_weights,
            'train_subset_size': train_subset_size,
            'split_size': split_size,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'verbose': verbose,
        }

        self._transformer = transformer

    def transform(self, X):
        return self._transformer.transform(X)
        
    def transformer(self):
        return self._transformer

    def knnEvaluationBuilder(self, X, y):
        def class_separation(X, labels):
            unique_labels, label_inds = np.unique(labels, return_inverse=True)
            ratio = 0
            for li in range(len(unique_labels)):
                Xc = X[label_inds==li]
                Xnc = X[label_inds!=li]
                ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc,Xnc).mean()
            return ratio / len(unique_labels)

        def knnEvaluation(individual):
            transformer = self._transformer.duplicate_instance().fit(self._input_dim, individual)

            subset = self.params['train_subset_size']
            train_mask = np.random.choice([True, False], X.shape[0], p=[subset, 1-subset])
            X_train, X_test, y_train, y_test = train_test_split(X[train_mask], y[train_mask], test_size=self.params['split_size'], random_state=self.params['random_state'])

            X_train_trans = transformer.transform(X_train)
            X_test_trans = transformer.transform(X_test)

            if self.params['classifier']=='svm':
                classifier = SVC(
                    random_state=self.params['random_state'])
            elif self.params['classifier']=='lsvm':
                classifier = LinearSVC(
                    dual=False,
                    random_state=self.params['random_state'])
            else: # == knn
                classifier = KNeighborsClassifier(
                    n_neighbors=self.params['n_neighbors'],
                    n_jobs=self.params['n_jobs'],
                    weights=self.params['knn_weights'])
            classifier.fit(X_train_trans, y_train)
            score = classifier.score(X_test_trans, y_test)

            # if self.params['class_separation']:
            separation_score = class_separation(X_test_trans, y_test)
            return [score, self.params['class_separation']*separation_score]
            # else:
            #     return [score]
            return [score - mean_squared_error(individual, np.ones(self._input_dim))]
            return [score - np.sum(np.absolute(individual))]
        
        return knnEvaluation

    def fit(self, X, Y):
        '''
         X: (n, d) array-like of samples
         Y: (n,) array-like of class labels
        '''
        np.random.seed(self.params['random_state'])
        self._input_dim = X.shape[1]

        sizeOfIndividual = self._transformer.individual_size(self._input_dim)
        
        strategy = cma.Strategy(centroid=[0.0]*sizeOfIndividual, sigma=1.0)
        toolbox.register("evaluate", self.knnEvaluationBuilder(X, Y))
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
        
        self._transformer.fit(self._input_dim, self.hof[0])
        return self
    
    def fit_transform(self, X, Y):
        self.fit(X,Y)
        return self.transform(X)
