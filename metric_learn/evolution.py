# The CMA-ES algorithm takes a population of one individual as argument
# See http://www.lri.fr/~hansen/cmaes_inmatlab.html
# for more details about the rastrigin and other tests for CMA-ES

from __future__ import division, absolute_import
import numpy as np

import itertools
import math
import random

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.base import ClassifierMixin
from scipy.spatial import distance

from deap import algorithms, base, benchmarks, cma, creator, tools
from concurrent.futures import ThreadPoolExecutor 
from .base_metric import BaseMetricLearner

# This DEAP settings needs to be global because of parallelism
creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

class BaseBuilder():
    def __init__(self):
        raise NotImplementedError('BaseBuilder should not be instantiated')

    def _get_extra_params(self, prefixes):
        params = {}
        for pname, pvalue in self.params.items():
            if '__' not in pname: continue

            prefix, pkey = pname.split('__', 1)
            if prefix not in prefixes: continue

            params[pkey] = pvalue

        return params

    def transformer_builder(self, transformer=None, params=None):
        if transformer is None:
            transformer = self.params['transformer']
        if params is None:
            params = self._get_extra_params(('transformer', 't'))

        def build():
            if isinstance(transformer, MatrixTransformer):
                return transformer
            elif transformer == 'diagonal':
                return DiagonalMatrixTransformer(**params)
            elif transformer == 'full':
                return FullMatrixTransformer(**params)
            elif transformer == 'neuralnetwork':
                return NeuralNetworkTransformer(**params)
            elif transformer == 'kmeans':
                return KMeansTransformer(**params)
            
            raise ValueError('Invalid `transformer` parameter value.')

        return build

    def _build_classifier(self, classifier):
        params = self._get_extra_params(('classifier', 'c'))

        if isinstance(classifier, ClassifierMixin):
            return classifier
        elif classifier == 'svc':
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

        raise ValueError('Invalid `classifier` parameter value.')

    def build_strategy(self, strategy, fitnesses, n_dim, transformer_builder):
        params = self._get_extra_params(('strategy', 's'))
        params.update({
            'fitnesses': fitnesses,
            'n_dim': n_dim,
            'transformer_builder': transformer_builder,
            'random_state': self.params['random_state'],
            'verbose': self.params['verbose'],
        })

        if isinstance(strategy, BaseEvolutionStrategy):
            return strategy
        elif strategy == 'cmaes':
            return CMAES(**params)
        elif strategy == 'de':
            return DifferentialEvolution(**params)
        elif strategy == 'dde':
            return DynamicDifferentialEvolution(**params)
        
        raise ValueError('Invalid `strategy` parameter value.')

    def build_fitnesses(self, fitnesses):
        if not isinstance(fitnesses, (list, tuple)):
            fitnesses = [fitnesses]

        return list(map(self.build_fitness, fitnesses))

    def build_fitness(self, fitness):
        def f(X_train, X_test, y_train, y_test):
            classifier = self._build_classifier(fitness)
            classifier.fit(X_train, y_train)
            return classifier.score(X_test, y_test)

        if fitness in ('knn', 'svc', 'lsvc'):
            return f

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
            raise Error('Invalid size of input_dim')

        self.L = np.diag(flat_weights)
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
            raise Error('input_dim and flat_weights sizes do not match')

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

class KMeansTransformer(MatrixTransformer, BaseBuilder):
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
            self._transformer = self.transformer_builder()()

        return self._transformer.individual_size(input_dim)

    def fit(self, X, y, flat_weights):
        self._transformer = self.transformer_builder()()
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
    def __init__(self, n_dim, fitnesses, transformer_builder=None, n_gen=25, split_size=0.33, train_subset_size=1.0, random_state=None, verbose=False):
        self.params = {
            'n_dim': n_dim,
            'fitnesses': fitnesses,
            'transformer_builder': transformer_builder,
            'n_gen': n_gen,
            'split_size': split_size,
            'train_subset_size': train_subset_size,
            'random_state': random_state,
            'verbose': verbose,
        }

    def fit(self, X, y, flat_weights):
        raise NotImplementedError('fit() is not implemented')

    def best_individual(self):
        raise NotImplementedError('best_individual() is not implemented')
    
    def _build_stats(self, verbose):
        if verbose == False:
            return None

        fitness = tools.Statistics(key=lambda ind: ind.fitness.values)
        fitness.register("avg", np.mean, axis=0)
        fitness.register("std", np.std, axis=0)
        fitness.register("min", np.min, axis=0)
        fitness.register("max", np.max, axis=0)

        return fitness
        # stats_size = tools.Statistics()
        # stats_size.register("x", lambda x: x)
        # stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)

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

    def evaluation_builder(self, X, y):
        # def class_separation(X, y):
        #     unique_labels, label_inds = np.unique(y, return_inverse=True)
        #     ratio = 0
        #     for li in range(len(unique_labels)):
        #         Xc = X[label_inds==li]
        #         Xnc = X[label_inds!=li]
        #         ratio += pairwise_distances(Xc).mean() / pairwise_distances(Xc,Xnc).mean()
        #     return ratio / len(unique_labels)

        def evaluate(individual):
            X_train, X_test, y_train, y_test = self._subset_train_test_split(X, y)

            if self.params['transformer_builder']:
                transformer = self.params['transformer_builder']()
                transformer.fit(X_train, y_train, individual)
                X_train = transformer.transform(X_train)
                X_test = transformer.transform(X_test)

            return [f(X_train, X_test, y_train, y_test) for f in self.params['fitnesses']]

            # classifier = self._build_classifier(self.params['classifier'])
            # classifier.fit(X_train_trans, y_train)
            # score = classifier.score(X_test_trans, y_test)

            # if self.params['class_separation']:
            #     return (score, class_separation(X_test_trans, y_test),)
            # else:
            #     return (score, 0,)

            # return [score, self.params['class_separation']*separation_score]
            # return [score - mean_squared_error(individual, np.ones(self._input_dim))]
            # return [score - np.sum(np.absolute(individual))]
        return evaluate

class CMAES(BaseEvolutionStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.params.update({
            # 'n_dim': n_dim,
        })

    def best_individual(self):
        return self.hall_of_fame[0]

    def fit(self, X, y):
        strategy = cma.Strategy(centroid=[0.0]*self.params['n_dim'], sigma=1.0)

        toolbox = base.Toolbox()
        toolbox.register("map", ThreadPoolExecutor(max_workers=None).map)
        
        toolbox.register("evaluate", self.evaluation_builder(X, y))
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        self.hall_of_fame = tools.HallOfFame(1)

        self.pop, self.logbook = algorithms.eaGenerateUpdate(
            toolbox,
            ngen=self.params['n_gen'],
            stats=self._build_stats(self.params['verbose']),
            halloffame=self.hall_of_fame,
            verbose=self.params['verbose']
        )

        return self


class DifferentialEvolution(BaseEvolutionStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.params.update({
            # 'n_dim': n_dim,
        })

    def best_individual(self):
        return self.hall_of_fame[0]

    def fit(self, X, y):
        individual_size = self.params['n_dim']
        
        toolbox = base.Toolbox()
        toolbox.register("map", ThreadPoolExecutor(max_workers=None).map)
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, individual_size)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selRandom, k=3)
        
        toolbox.register("evaluate", self.evaluation_builder(X, y))

        self.hall_of_fame = tools.HallOfFame(1)
        stats = self._build_stats(self.params['verbose'])

        # Differential evolution parameters
        CR = 0.25
        F = 1  
        MU = 50
        
        pop = toolbox.population(n=MU);
        
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        
        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        
        if stats:
            record = stats.compile(pop)
            logbook.record(gen=0, evals=len(pop), **record)
            print(logbook.stream)
        
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
                logbook.record(gen=g, evals=len(pop), **record)
                print(logbook.stream)

        return self


class DynamicDifferentialEvolution(BaseEvolutionStrategy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.params.update({
            # 'n_dim': n_dim,
        })

    def best_individual(self):
        return self.hall_of_fame[0]

    def _brown_ind(self, iclass, best, sigma):
            return iclass(random.gauss(x, sigma) for x in best)

    def fit(self, X, y):
        # Differential evolution parameters
        NPOP = 10 # Should be equal to the number of peaks
        CR = 0.6
        F = 0.4
        regular, brownian = 4, 2
        BOUNDS = (-1, 1)

        individual_size = self.params['n_dim']

        toolbox = base.Toolbox()
        toolbox.register("map", ThreadPoolExecutor(max_workers=None).map)
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, individual_size)
        toolbox.register("brownian_individual", self._brown_ind, creator.Individual, sigma=0.3)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register("select", random.sample, k=4)
        toolbox.register("best", tools.selBest, k=1)

        toolbox.register("evaluate", self.evaluation_builder(X, y))

        self.hall_of_fame = tools.HallOfFame(1)
        stats = self._build_stats(self.params['verbose'])

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        
        # Initialize populations
        populations = [toolbox.population(n=regular + brownian) for _ in range(NPOP)]

        # Evaluate the individuals
        for idx, subpop in enumerate(populations):
            fitnesses = toolbox.map(toolbox.evaluate, subpop)
            for ind, fit in zip(subpop, fitnesses):
                ind.fitness.values = fit

        if stats:
            record = stats.compile(itertools.chain(*populations))
            logbook.record(gen=0, evals=len(populations), **record)
            print(logbook.stream)

        for g in range(1, self.params['n_gen']):
            # Detect a change and invalidate fitnesses if necessary
            bests = [toolbox.best(subpop)[0] for subpop in populations]
            if any(b.fitness.values != toolbox.evaluate(b) for b in bests):
                for individual in itertools.chain(*populations):
                    del individual.fitness.values

            # Apply exclusion
            rexcl = (BOUNDS[1] - BOUNDS[0]) / (2 * NPOP**(1.0/individual_size))
            for i, j in itertools.combinations(range(NPOP), 2):
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
                logbook.record(gen=g, evals=len(populations), **record)
                print(logbook.stream)

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
    '''
    CMAES
    '''
    def __init__(self, strategy='cmaes', fitnesses='knn', transformer='full',
                 random_state=None, verbose=False, **kwargs):
        """Initialize the learner.

        Parameters
        ----------
        fitnesses : ('knn', 'svc', 'lsvc', fitnesses object)
            fitnesses is used in fitness scoring
        transformer : ('full', 'diagonal', MatrixTransformer object)
            transformer defines transforming function to learn
        verbose : bool, optional
            if True, prints information while learning
        """
        self.params = {
            **kwargs,
            'strategy': strategy,
            'fitnesses': fitnesses,
            'transformer': transformer,
            'random_state': random_state,
            'verbose': verbose,
        }
        np.random.seed(random_state)

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
        transformer_builder = self.transformer_builder()
        self._transformer = transformer_builder()
        
        strategy = self.build_strategy(
            strategy=self.params['strategy'],
            fitnesses=self.build_fitnesses(self.params['fitnesses']),
            n_dim=self._transformer.individual_size(X.shape[1]),
            transformer_builder=transformer_builder,
        )
        strategy.fit(X, y)
        
        self._transformer.fit(X, y, strategy.best_individual())
        return self
