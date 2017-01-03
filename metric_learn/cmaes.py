from __future__ import division, absolute_import
import numpy as np
from concurrent.futures import ThreadPoolExecutor 

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from deap import algorithms, base, benchmarks, cma, creator, tools

from .base_metric import BaseMetricLearner
from .evo_metric import EvoMetric

class CMAES(BaseMetricLearner):
    def __init__(self, metric='diagonal', n_gen=250, n_neighbors=1, knn_weights='uniform', split_size=0.33, n_jobs=-1, verbose=False):
        if metric not in ('diagonal', 'full'):
            raise ValueError('Invalid metric: %r' % metric)

        self.params = {
            'metric': metric,
            'n_gen': n_gen,
            'n_neighbors': n_neighbors,
            'knn_weights': knn_weights,
            'split_size': split_size,
            'n_jobs': n_jobs,
            'verbose': verbose,
        }

    def transform(self, X):
        return X.dot(self.transformer().T)
        
    def transformer(self):
        return self.L

    def knnEvaluationBuilder(self, X, y, N):
        assert N == X.shape[1]
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=47)
        
#         def knnEvaluation(individual):
#             es = EvoMetric(individual, N)
            
#             subset = .1
#             train_mask = np.random.choice([True, False], X.shape[0], p=[subset, 1-subset])
#             X_train, X_test, y_train, y_test = train_test_split(X[train_mask], y[train_mask], test_size=0.33)#, random_state=47)
#             X_train_trans = es.transform(X_train)
#             X_test_trans = es.transform(X_test)
#             knn = KNeighborsClassifier(n_neighbors=8, n_jobs=-1)
#             knn.fit(X_train_trans, y_train)
#             score = knn.score(X_test_trans, y_test)

#             return [score]
#             return [score - mean_squared_error(individual, np.ones(N))]
#             return [score - np.sum(np.absolute(individual))]
    
        def knnEvaluation(individual):
            es = EvoMetric(individual, N)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.params['split_size'])#, random_state=47)
            X_train_trans = es.transform(X_train)
            X_test_trans = es.transform(X_test)
            knn = KNeighborsClassifier(
                n_neighbors=self.params['n_neighbors'],
                n_jobs=self.params['n_jobs'],
                weights=self.params['knn_weights'])
            knn.fit(X_train_trans, y_train)
            score = knn.score(X_test_trans, y_test)

            return [score]
            return [score - mean_squared_error(individual, np.ones(N))]
            return [score - np.sum(np.absolute(individual))]
        
        return knnEvaluation

    def fit(self, X, Y):
        '''
         X: (n, d) array-like of samples
         Y: (n,) array-like of class labels
        '''
        self.N = X.shape[1]
        
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("evaluate", self.knnEvaluationBuilder(X, Y, self.N))
        toolbox.register("map", ThreadPoolExecutor(max_workers=None).map)
        
        # The cma module uses the numpy random number generator
        np.random.seed(128)

        # The CMA-ES algorithm takes a population of one individual as argument
        # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
        # for more details about the rastrigin and other tests for CMA-ES
        
        if self.params['metric'] == 'diagonal':
            sizeOfIndividual = self.N
        else:
            sizeOfIndividual = self.N**2
        
        strategy = cma.Strategy(centroid=[0.0]*sizeOfIndividual, sigma=10.0) # lambda_=20*N
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        self.hof = tools.HallOfFame(1)
        
        if self.params['verbose']:
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)
        else:
            stats=None
#         logger = tools.EvolutionLogger(stats.functions.keys())

        # The CMA-ES algorithm converge with good probability with those settings
        pop, logbook = algorithms.eaGenerateUpdate(
            toolbox,
            ngen=self.params['n_gen'],
            stats=stats,
            halloffame=self.hof,
            verbose=self.params['verbose']
        )
        
        if self.params['metric'] == 'diagonal':
            self.L = np.diag(self.hof[0])
        else:
            self.L = np.reshape(self.hof[0], (self.N, self.N))
        return self
    
    def fit_transform(self, X, Y):
        self.fit(X,Y)
        return self.transform(X)
