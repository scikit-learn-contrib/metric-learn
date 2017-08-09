# The CMA-ES algorithm takes a population of one individual as argument
# See http://www.lri.fr/~hansen/cmaes_inmatlab.html
# for more details about the rastrigin and other tests for CMA-ES

from __future__ import division, absolute_import
import numpy as np
import scipy

import itertools
import math
import random

from sklearn.base import ClassifierMixin, clone
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from scipy.spatial import distance


from deap import algorithms, base, cma, tools
from concurrent.futures import ThreadPoolExecutor 
from .base_metric import BaseMetricLearner

class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = None

class MultidimensionalFitness(base.Fitness):
    def __init__(self, n_dim, *args, **kwargs):
        self.n_dim = n_dim
        self.weights = (1.0,)*n_dim
        
        super().__init__(*args, **kwargs)

    def __deepcopy__(self, memo):
        copy_ = self.__class__(self.n_dim)
        copy_.wvalues = self.wvalues
        return copy_

class BaseFitness():
    def __init__(self):
        pass

    @staticmethod
    def available(method):
        return False

    def __call__(self, X_train, X_test, y_train, y_test):
        raise NotImplementedError('__call__ has not been implemented')

class ScorerFitness(BaseFitness):
    def __init__(self, classifier, **kwargs):
        self.params = {
            'classifier': classifier,
        }
        self.classifier_params = kwargs

    @staticmethod
    def available(method):
        return (method in ['knn', 'scv', 'lsvc']) or \
                isinstance(method, ClassifierMixin)

    def __call__(self, X_train, X_test, y_train, y_test):
        classifier = self._build_classifier(
            self.params['classifier'],
            self.classifier_params,
        )
        classifier.fit(X_train, y_train)
        return classifier.score(X_test, y_test)
        # return [score - mean_squared_error(individual, np.ones(self._input_dim))]
        # return [score - np.sum(np.absolute(individual))]

    def _build_classifier(self, classifier, params):
        if isinstance(classifier, ClassifierMixin):
            return clone(classifier)
        elif classifier == 'svc':
            return SVC(**params)
        elif classifier == 'lsvc':
            return LinearSVC(**params)
        elif classifier == 'knn':
            return KNeighborsClassifier(**params)

        raise ValueError('Invalid `classifier` parameter value.')

class RandomFitness(BaseFitness):
    @staticmethod
    def available(method):
        return method in ['random']

    def __call__(self, X_train, X_test, y_train, y_test):
        return np.random.random()

class WeightedPurityFitness(BaseFitness):
    def __init__(self, sig=5, kmeans__n_init=1):
        self.params = {
            'sig': sig,
            'kmeans__n_init': kmeans__n_init,
        }

    @staticmethod
    def available(method):
        return method in ['wpur']

    def __call__(self, X_train, X_test, y_train, y_test):
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        le = LabelEncoder()
        y = le.fit_transform(y)

        kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=self.params['kmeans__n_init'])
        kmeans.fit(X)

        r = distance.cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
        h = np.exp(-r/(self.params['sig']**2))

        N = confusion_matrix(y, kmeans.labels_)

        wN = np.zeros(h.shape)
        for l in range(wN.shape[0]): # label
            for c in range(wN.shape[0]): # cluster
                for j in range(wN.shape[0]):
                    wN[l,c] += h[l,c]*N[l,j]

        return wN.max(axis=0).sum() / wN.sum()

class WeightedFMeasureFitness(BaseFitness):
    def __init__(self, weighted=False, sig=5, kmeans__n_init=1):
        self.params = {
            'weighted': weighted,
            'sig': sig,
            'kmeans__n_init': kmeans__n_init,
        }

    @staticmethod
    def available(method):
        return method in ['wfme']

    def __call__(self, X_train, X_test, y_train, y_test):
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        le = LabelEncoder()
        y = le.fit_transform(y)

        kmeans = KMeans(n_clusters=len(np.unique(y)), n_init=self.params['kmeans__n_init'])
        kmeans.fit(X)

        if not self.params['weighted']:
            return f1_score(y, kmeans.labels_, average='weighted')

        r = distance.cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
        h = np.exp(-r/(self.params['sig']**2))

        N = confusion_matrix(y, kmeans.labels_)

        wN = np.zeros(h.shape)
        for l in range(wN.shape[0]): # label
            for c in range(wN.shape[0]): # cluster
                for j in range(wN.shape[0]):
                    wN[l,c] += h[l,c]*N[l,j]

        Prec = wN / wN.sum(axis=0)
        Rec  = wN /  N.sum(axis=1)[:,None]
        F    = (2*Prec*Rec) / (Prec+Rec)

        wFme = 0
        for l in range(F.shape[0]):
            wFme += (N[l,:].sum()/N.sum())*F[l,:].max()

        return wFme

class ClassSeparationFitness(BaseFitness):
    @staticmethod
    def available(method):
        return method in ['class_separation']

    def __call__(self, X_train, X_test, y_train, y_test):
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        unique_labels, label_inds = np.unique(y, return_inverse=True)
        ratio = 0
        for li in range(len(unique_labels)):
            Xc = X[label_inds==li]
            Xnc = X[label_inds!=li]
            ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc,Xnc).mean()
        return -ratio / len(unique_labels)

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

    def build_transformer(self, transformer_shape=None, params={}):
        if transformer_shape is None:
            transformer_shape = self.transformer_shape
        # if params is None:
            # params = self._get_extra_params(('transformer_shape', 't'))

        if isinstance(transformer_shape, MatrixTransformer):
            return transformer_shape
        elif transformer_shape == 'diagonal':
            return DiagonalMatrixTransformer(**params)
        elif transformer_shape == 'full':
            return FullMatrixTransformer(**params)
        elif transformer_shape == 'triangular':
            return TriangularMatrixTransformer(**params)
        elif transformer_shape == 'neuralnetwork':
            return NeuralNetworkTransformer(**params)
        elif transformer_shape == 'kmeans':
            return KMeansTransformer(**params)
        
        raise ValueError('Invalid `transformer_shape` parameter value: `{}`'.format(transformer_shape))

class MatrixTransformer(BaseMetricLearner):
    def __init__(self):
        raise NotImplementedError('MatrixTransformer should not be instantiated')

    def duplicate_instance(self):
        return self.__class__(**self.params)

    def individual_size(self, input_dim):
        raise NotImplementedError('individual_size() is not implemented')

    def fit(self, X, y, flat_weights):
        raise NotImplementedError('fit() is not implemented')

    def transform(self, X):
        return X.dot(self.transformer().T)
        
    def transformer(self):
        return self.L

class DiagonalMatrixTransformer(MatrixTransformer):
    def __init__(self):
        self.params = {}

    def individual_size(self, input_dim):
        return input_dim

    def fit(self, X, y, flat_weights):
        self.input_dim = X.shape[1]
        if self.input_dim != len(flat_weights):
            raise Exception('`input_dim` and `flat_weights` sizes do not match: {} vs {}'
                .format(input_dim, len(flat_weights)))

        self.L = np.diag(flat_weights)
        return self

class TriangularMatrixTransformer(MatrixTransformer):
    def __init__(self):
        self.params = {}

    def individual_size(self, input_dim):
        return input_dim*(input_dim+1)//2

    def fit(self, X, y, flat_weights):
        input_dim = X.shape[1]
        if self.individual_size(input_dim) != len(flat_weights):
            raise Exception('`input_dim` and `flat_weights` sizes do not match: {} vs {}'
                .format(self.individual_size(input_dim), len(flat_weights)))

        self.L = np.zeros((input_dim, input_dim))
        self.L[np.tril_indices(input_dim, 0)] = flat_weights
        return self

class FullMatrixTransformer(MatrixTransformer):
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
            raise Exception('`input_dim` and `flat_weights` sizes do not match: {} vs {}'
                .format(self.individual_size(input_dim), len(flat_weights)))

        self.L = np.reshape(flat_weights, (len(flat_weights)//input_dim, input_dim))
        return self

class NeuralNetworkTransformer(MatrixTransformer):
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
        elif activation == 'sigm':
            return scipy.special.expit
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
            raise Exception('Invalid size of the flat_weights')

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

class KMeansTransformer(MatrixTransformer, BaseBuilder):
    def __init__(self, transformer='full', n_clusters='classes', function='distance', n_init=1, random_state=None, **kwargs):
        self._transformer = None
        self.params = {            
            'transformer': transformer,
            'n_clusters': n_clusters,
            'function': function,
            'n_init': n_init,
            'random_state': random_state,
        }
        self.params.update(**kwargs)

    def individual_size(self, input_dim):
        if self._transformer is None:
            self._transformer = self.build_transformer()

        return self._transformer.individual_size(input_dim)

    def fit(self, X, y, flat_weights):
        self._transformer = self.build_transformer()
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

class BaseEvolutionStrategy():
    def __init__(self, n_dim, fitnesses, transformer=None, n_gen=25, split_size=0.33, train_subset_size=1.0, stats=None, random_state=None, verbose=False):
        self.params = {
            'n_dim': n_dim,
            'fitnesses': fitnesses,
            'transformer': transformer,
            'n_gen': n_gen,
            'split_size': split_size,
            'train_subset_size': train_subset_size,
            'stats': stats,
            'random_state': random_state,
            'verbose': verbose,
        }

    def fit(self, X, y, flat_weights):
        raise NotImplementedError('fit() is not implemented')

    def best_individual(self):
        raise NotImplementedError('best_individual() is not implemented')
    
    def _build_stats(self):
        if self.params['stats'] is None:
            return None
        elif isinstance(self.params['stats'], tools.Statistics):
            return self.params['stats']
        elif self.params['stats'] == 'identity':
            fitness = tools.Statistics(key=lambda ind: ind)
            fitness.register("id", lambda ind: ind)
            return fitness

        fitness = tools.Statistics(key=lambda ind: ind.fitness.values)
        fitness.register("avg", np.mean, axis=0)
        fitness.register("std", np.std, axis=0)
        fitness.register("min", np.min, axis=0)
        fitness.register("max", np.max, axis=0)
        return fitness

    def _subset_train_test_split(self, X, y):
        subset = self.params['train_subset_size']
        assert(0.0 < subset <= 1.0)

        if subset==1.0:
            return train_test_split(X, y, 
                test_size=self.params['split_size'],
                random_state=self.params['random_state'],
            )

        train_mask = np.random.choice([True, False], X.shape[0], p=[subset, 1-subset])
        return train_test_split(X[train_mask], y[train_mask],
            test_size=self.params['split_size'],
            random_state=self.params['random_state'],
        )

    def generate_individual_with_fitness(self, func, n):
        fitness_len = len(self.params['fitnesses'])
        ind = Individual(func() for _ in range(n))
        ind.fitness = MultidimensionalFitness(fitness_len)
        return ind

    def create_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("map", ThreadPoolExecutor(max_workers=None).map)

        return toolbox

    def cut_individual(self, individual):
        return individual

    def evaluation_builder(self, X, y):
        def evaluate(individual):
            X_train, X_test, y_train, y_test = self._subset_train_test_split(X, y)

            # transform the inputs if there is a transformer
            if self.params['transformer']:
                transformer = self.params['transformer'].duplicate_instance()
                transformer.fit(X_train, y_train, self.cut_individual(individual))
                X_train = transformer.transform(X_train)
                X_test = transformer.transform(X_test)

            return [f(X_train, X_test, y_train, y_test) for f in self.params['fitnesses']]

        return evaluate

class CMAES(BaseEvolutionStrategy):
    def __init__(self, mean=0.0, sigma=1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.params.update({
            'mean': mean,
            'sigma': sigma,
        })

    def best_individual(self):
        return self.hall_of_fame[0]

    def _generate_pop_with_fitness(self, generate):
        fitness_len = len(self.params['fitnesses'])
        
        individuals = generate(Individual)
        for ind in individuals:
            setattr(ind, 'fitness', MultidimensionalFitness(fitness_len))

        return individuals

    def fit(self, X, y):
        strategy = cma.Strategy(
            centroid=[self.params['mean']]*self.params['n_dim'],
            sigma=self.params['sigma'],
        )
        
        toolbox = self.create_toolbox()
        toolbox.register("evaluate", self.evaluation_builder(X, y))
        toolbox.register("generate", self._generate_pop_with_fitness, strategy.generate)
        toolbox.register("update", strategy.update)

        self.hall_of_fame = tools.HallOfFame(1)

        self.pop, self.logbook = algorithms.eaGenerateUpdate(
            toolbox,
            ngen=self.params['n_gen'],
            stats=self._build_stats(),
            halloffame=self.hall_of_fame,
            verbose=self.params['verbose']
        )

        return self


class DifferentialEvolution(BaseEvolutionStrategy):
    def __init__(self, population_size=50, cr=.25, f=1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.params.update({
            'population_size': population_size,
            'cr': cr,
            'f': f,
        })

    def best_individual(self):
        return self.hall_of_fame[0]

    def fit(self, X, y):
        individual_size = self.params['n_dim']
        
        toolbox = self.create_toolbox()
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register("individual", self.generate_individual_with_fitness, toolbox.attr_float, individual_size)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selRandom, k=3)
        
        toolbox.register("evaluate", self.evaluation_builder(X, y))

        self.hall_of_fame = tools.HallOfFame(1)
        stats = self._build_stats()

        CR = self.params['cr']
        F = self.params['f']
        pop = toolbox.population(n=self.params['population_size'])
        
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        if stats:
            record = stats.compile(pop)
            self.logbook.record(gen=0, evals=len(pop), **record)
            if self.params['verbose']: print(self.logbook.stream)
        
        for g in range(1, self.params['n_gen']):
            for k, agent in enumerate(pop):
                a,b,c = toolbox.select(pop)
                y = toolbox.clone(agent)
                index = np.random.randint(individual_size)
                for i, value in enumerate(agent):
                    if i == index or np.random.random() < CR:
                        y[i] = a[i] + F*(b[i]-c[i])
                y.fitness.values = toolbox.evaluate(y)
                if y.fitness > agent.fitness:
                    pop[k] = y
            self.hall_of_fame.update(pop)
            
            if stats:
                record = stats.compile(pop)
                self.logbook.record(gen=g, evals=len(pop), **record)
                if self.params['verbose']: print(self.logbook.stream)

        return self

class SelfAdaptingDifferentialEvolution(BaseEvolutionStrategy):
    def __init__(self, population_size=None, Fl=0.1, Fu=0.9, t1=0.1, t2=0.1, **kwargs):
        super().__init__(**kwargs)
        
        self.params.update({
            'population_size': population_size,
            'Fl': Fl,
            'Fu': Fu,
            't1': t1,
            't2': t2,
        })

    def cut_individual(self, individual):
        return individual[2:]

    def best_individual(self):
        return self.cut_individual(self.hall_of_fame[0])

    def fit(self, X, y):
        individual_size = self.params['n_dim']+2
        
        toolbox = self.create_toolbox()
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register("individual", self.generate_individual_with_fitness, toolbox.attr_float, individual_size)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selRandom, k=3)
        toolbox.register("evaluate", self.evaluation_builder(X, y))

        self.hall_of_fame = tools.HallOfFame(1)
        stats = self._build_stats()

        # TODO: Make this more general and move to BaseEvolutionStrategy
        if self.params['population_size'] == 'log':
            population_size = int(4 + 3 * np.log(self.params['n_dim']))
        elif self.params['population_size'] is not None:
            population_size = self.params['population_size']
        else:
            population_size = 10*self.params['n_dim']
        pop = toolbox.population(n=population_size)
        
        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        if stats:
            record = stats.compile(pop)
            self.logbook.record(gen=0, evals=len(pop), **record)
            if self.params['verbose']: print(self.logbook.stream)
        
        for g in range(1, self.params['n_gen']):
            for k, agent in enumerate(pop):
                a,b,c = toolbox.select(pop)
                y = toolbox.clone(agent)

                # Update the control parameters
                if np.random.random() < self.params['t1']: # F
                    y[0] = self.params['Fl'] + np.random.random()*self.params['Fu']
                if np.random.random() < self.params['t2']: # CR
                    y[1] = np.random.random()

                # Mutation and crossover
                index = np.random.randint(2, individual_size)
                for i, value in enumerate(agent[2:], 2):
                    if i == index or np.random.random() < y[1]:
                        y[i] = a[i] + y[0]*(b[i]-c[i])
                
                # Selection
                y.fitness.values = toolbox.evaluate(y)
                if y.fitness >= agent.fitness:
                    pop[k] = y
            self.hall_of_fame.update(pop)
            
            if stats:
                record = stats.compile(pop)
                self.logbook.record(gen=g, evals=len(pop), **record)
                if self.params['verbose']: print(self.logbook.stream)

        return self


class DynamicDifferentialEvolution(BaseEvolutionStrategy):
    def __init__(self, population_size=10, population_regular=4, population_brownian=2, cr=0.6, f=0.4, bounds=(-1.0, 1.0), **kwargs):
        super().__init__(**kwargs)
        
        self.params.update({
            'population_size': population_size,
            'cr': cr,
            'f': f,
            'bounds': bounds,
            'population_regular': population_regular,
            'population_brownian': population_brownian,
        })

    def best_individual(self):
        return self.hall_of_fame[0]

    def generate_brow_ind_with_fitness(self, best, sigma=0.3):
        fitness_len = len(self.params['fitnesses'])
        ind = Individual(random.gauss(x, sigma) for x in best)
        ind.fitness = MultidimensionalFitness(fitness_len)
        return ind

    def fit(self, X, y):
        # Differential evolution parameters
        individual_size = self.params['n_dim']
        population_size = self.params['population_size'] # Should be equal to the number of peaks
        
        CR, F = self.params['cr'], self.params['f']
        regular, brownian = self.params['population_regular'], self.params['population_brownian']
        bounds = self.params['bounds']

        toolbox = self.create_toolbox()
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register("individual", self.generate_individual_with_fitness, toolbox.attr_float, individual_size)
        toolbox.register("brownian_individual", self.generate_brow_ind_with_fitness)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("select", random.sample, k=4)
        toolbox.register("best", tools.selBest, k=1)

        toolbox.register("evaluate", self.evaluation_builder(X, y))

        self.hall_of_fame = tools.HallOfFame(1)
        stats = self._build_stats()

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
        # Initialize populations
        populations = [toolbox.population(n=regular + brownian) for _ in range(population_size)]

        # Evaluate the individuals
        for idx, subpop in enumerate(populations):
            fitnesses = toolbox.map(toolbox.evaluate, subpop)
            for ind, fit in zip(subpop, fitnesses):
                ind.fitness.values = fit

        if stats:
            record = stats.compile(itertools.chain(*populations))
            self.logbook.record(gen=0, evals=len(populations), **record)
            if self.params['verbose']: print(self.logbook.stream)

        for g in range(1, self.params['n_gen']):
            # Detect a change and invalidate fitnesses if necessary
            bests = [toolbox.best(subpop)[0] for subpop in populations]
            if any(b.fitness.values != toolbox.evaluate(b) for b in bests):
                for individual in itertools.chain(*populations):
                    del individual.fitness.values

            # Apply exclusion
            rexcl = (bounds[1] - bounds[0]) / (2 * population_size**(1.0/individual_size))
            for i, j in itertools.combinations(range(population_size), 2):
                if bests[i].fitness.valid and bests[j].fitness.valid:
                    d = sum((bests[i][k] - bests[j][k])**2 for k in range(individual_size))
                    d = math.sqrt(d)

                    if d < rexcl:
                        if bests[i].fitness < bests[j].fitness:
                            k = i
                        else:
                            k = j

                        populations[k] = toolbox.population(n=regular + brownian)
            
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in itertools.chain(*populations) if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            all_pops = list(itertools.chain(*populations))
            self.hall_of_fame.update(all_pops)
        
            if stats:
                record = stats.compile(all_pops)
                self.logbook.record(gen=g, evals=len(populations), **record)
                if self.params['verbose']: print(self.logbook.stream)

            # Evolve the sub-populations
            for idx, subpop in enumerate(populations):
                newpop = []
                xbest, = toolbox.best(subpop)
                # Apply regular DE to the first part of the population
                for individual in subpop[:regular]:
                    x1, x2, x3, x4 = toolbox.select(subpop)
                    offspring = toolbox.clone(individual)
                    index = np.random.randint(individual_size)
                    for i, value in enumerate(individual):
                        if i == index or np.random.random() < CR:
                            offspring[i] = xbest[i] + F * (x1[i] + x2[i] - x3[i] - x4[i])
                    offspring.fitness.values = toolbox.evaluate(offspring)
                    if offspring.fitness >= individual.fitness:
                        newpop.append(offspring)
                    else:
                        newpop.append(individual)

                # Apply Brownian to the last part of the population
                newpop.extend(toolbox.brownian_individual(xbest) for _ in range(brownian))

                # Evaluate the brownian individuals
                for individual in newpop[-brownian:]:
                    individual.fitness.value = toolbox.evaluate(individual)

                # Replace the population 
                populations[idx] = newpop
        
        return self

class MetricEvolution(BaseMetricLearner, BaseBuilder):
    def __init__(self, strategy='cmaes', fitnesses='knn', transformer_shape='full',
                 random_state=None, verbose=False):
        """Initialize the learner.

        Parameters
        ----------
        fitnesses : ('knn', 'svc', 'lsvc', fitnesses object)
            fitnesses is used in fitness scoring
        transformer_shape : ('full', 'diagonal', MatrixTransformer object)
            transformer shape defines transforming function to learn
        verbose : bool, optional
            if True, prints information while learning
        """
        self.strategy = strategy
        self.fitnesses = fitnesses
        self.transformer_shape = transformer_shape
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

        if RandomFitness.available(fitness):
            return RandomFitness()

        if ScorerFitness.available(fitness):
            return ScorerFitness(fitness, **params)

        if ClassSeparationFitness.available(fitness):
            return ClassSeparationFitness()

        if WeightedPurityFitness.available(fitness):
            return WeightedPurityFitness(**params)

        if WeightedFMeasureFitness.available(fitness):
            return WeightedFMeasureFitness(**params)

         # TODO unify error messages
        raise ValueError('Invalid value of fitness: `{}`'.format(fitness))

    def build_strategy(self, strategy, strategy_params, fitnesses, n_dim, transformer, random_state, verbose):
        strategy_params = {
            'fitnesses': fitnesses,
            'n_dim': n_dim,
            'transformer': transformer,
            'random_state': random_state,
            'verbose': verbose,
        }

        if isinstance(strategy, BaseEvolutionStrategy):
            return strategy
        elif strategy == 'cmaes':
            return CMAES(**strategy_params)
        elif strategy == 'de':
            return DifferentialEvolution(**strategy_params)
        elif strategy == 'jde':
            return SelfAdaptingDifferentialEvolution(**strategy_params)
        elif strategy == 'dde':
            return DynamicDifferentialEvolution(**strategy_params)
        
        raise ValueError('Invalid `strategy` parameter value.')

    def transform(self, X):
        return self._transformer.transform(X)
        
    def transformer(self):
        return self._transformer

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def fit(self, X, y):
        '''
         X: (n, d) array-like of samples
         Y: (n,) array-like of class labels
        '''
        # Initialize Transformer builder and default transformer
        self._transformer = self.build_transformer()
        
        # Build strategy and fitnesses with correct params
        self._strategy = self.build_strategy(
            strategy = self.strategy,
            strategy_params = {}, # self._get_extra_params(('strategy', 's')),
            fitnesses = self.build_fitnesses(self.fitnesses),
            n_dim = self._transformer.individual_size(X.shape[1]),
            transformer = self._transformer,
            random_state = self.random_state,
            verbose = self.verbose,
        )

        # Evolve best transformer
        self._strategy.fit(X, y)

        # Fit transformer with the best individual
        self._transformer.fit(X, y, self._strategy.best_individual())
        return self
