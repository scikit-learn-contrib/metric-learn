from concurrent.futures import ThreadPoolExecutor

from deap import base, tools

import numpy as np

from sklearn.model_selection import train_test_split


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


class BaseEvolutionStrategy():
    def __init__(self, n_dim, fitnesses, transformer=None, n_gen=25,
                 split_size=0.33, train_subset_size=1.0, stats=None,
                 random_state=None, verbose=False):
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

        np.random.seed(random_state)

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

        if subset == 1.0:
            return train_test_split(
                X, y,
                test_size=self.params['split_size'],
                random_state=self.params['random_state'],
            )

        train_mask = np.random.choice(
            [True, False],
            X.shape[0],
            p=[subset, 1 - subset]
        )
        return train_test_split(
            X[train_mask], y[train_mask],
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
            X_train, X_test, y_train, y_test = self._subset_train_test_split(
                X, y,
            )

            # transform the inputs if there is a transformer
            if self.params['transformer']:
                transformer = self.params['transformer'].duplicate_instance()
                transformer.fit(
                    X_train,
                    y_train,
                    self.cut_individual(individual)
                )
                X_train = transformer.transform(X_train)
                X_test = transformer.transform(X_test)

            return [f(X_train, X_test, y_train, y_test)
                    for f in self.params['fitnesses']]

        return evaluate
