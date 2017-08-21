from concurrent.futures import ThreadPoolExecutor

from deap import base, tools

import numpy as np

from sklearn.model_selection import train_test_split

from .individual import Individual
from .mfitness import MultidimensionalFitness


class BaseEvolutionStrategy(object):
    def __init__(self, n_dim, fitnesses, transformer=None, n_gen=25,
                 split_size=0.33, train_subset_size=1.0, stats=None,
                 random_state=None, verbose=False):
        self.n_dim = n_dim
        self.fitnesses = fitnesses
        self.transformer = transformer
        self.n_gen = n_gen
        self.split_size = split_size
        self.train_subset_size = train_subset_size
        self.stats = stats
        self.random_state = random_state
        self.verbose = verbose

        # np.random.seed(random_state)

    def fit(self, X, y):
        raise NotImplementedError('fit() is not implemented')

    def best_individual(self):
        raise NotImplementedError('best_individual() is not implemented')

    def _build_stats(self):
        if self.stats is None:
            return None
        elif isinstance(self.stats, tools.Statistics):
            return self.stats
        elif self.stats == 'identity':
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
        subset = self.train_subset_size
        assert(0.0 < subset <= 1.0)

        if subset == 1.0:
            return train_test_split(
                X, y,
                test_size=self.split_size,
                random_state=self.random_state,
            )

        train_mask = np.random.choice(
            [True, False],
            X.shape[0],
            p=[subset, 1 - subset]
        )
        return train_test_split(
            X[train_mask], y[train_mask],
            test_size=self.split_size,
            random_state=self.random_state,
        )

    def generate_individual_with_fitness(self, func, n):
        fitness_len = len(self.fitnesses)
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
            if self.transformer:
                transformer = self.transformer.duplicate_instance()
                transformer.fit(
                    X_train,
                    y_train,
                    self.cut_individual(individual)
                )
                X_train = transformer.transform(X_train)
                X_test = transformer.transform(X_test)

            return [f(X_train, X_test, y_train, y_test)
                    for f in self.fitnesses]

        return evaluate
