from deap import tools

import numpy as np

from .base_strategy import BaseEvolutionStrategy


class SelfAdaptingDifferentialEvolution(BaseEvolutionStrategy):
    def __init__(self, population_size=None, fl=0.1, fu=0.9, t1=0.1, t2=0.1,
                 random_state=None, **kwargs):
        super().__init__(random_state=random_state, **kwargs)

        self.population_size = population_size
        self.fl = fl
        self.fu = fu
        self.t1 = t1
        self.t2 = t2

        np.random.seed(random_state)

    def cut_individual(self, individual):
        return individual[2:]

    def best_individual(self):
        return self.cut_individual(self.hall_of_fame[0])

    def fit(self, X, y):
        individual_size = self.n_dim + 2

        toolbox = self.create_toolbox()
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register(
            "individual",
            self.generate_individual_with_fitness,
            toolbox.attr_float,
            individual_size)
        toolbox.register(
            "population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluation_builder(X, y))

        self.hall_of_fame = tools.HallOfFame(1)
        stats = self._build_stats()

        # TODO: Make this more general and move to BaseEvolutionStrategy
        if self.population_size == 'log':
            population_size = int(4 + 3 * np.log(self.n_dim))
        elif self.population_size is not None:
            population_size = self.population_size
        else:
            population_size = 10 * self.n_dim
        pop = toolbox.population(n=population_size)

        self.logbook = tools.Logbook()
        self.logbook.header = ['gen', 'nevals'] \
            + (stats.fields if stats else [])

        # Evaluate the individuals
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        if stats:
            record = stats.compile(pop)
            self.logbook.record(gen=0, evals=len(pop), **record)
            if self.verbose:
                print(self.logbook.stream)

        for g in range(1, self.n_gen):
            for k, agent in enumerate(pop):
                idxs = np.random.choice(len(pop), size=3)
                a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
                y = toolbox.clone(agent)

                # Update the control parameters
                if np.random.random() < self.t1:  # F
                    y[0] = self.fl \
                        + np.random.random() * self.fu
                if np.random.random() < self.t2:  # CR
                    y[1] = np.random.random()

                # Mutation and crossover
                index = np.random.randint(2, individual_size)
                for i, value in enumerate(agent[2:], 2):
                    if i == index or np.random.random() < y[1]:
                        y[i] = a[i] + y[0] * (b[i] - c[i])

                # Selection
                y.fitness.values = toolbox.evaluate(y)
                if y.fitness >= agent.fitness:
                    pop[k] = y
            self.hall_of_fame.update(pop)

            if stats:
                record = stats.compile(pop)
                self.logbook.record(gen=g, evals=len(pop), **record)
                if self.verbose:
                    print(self.logbook.stream)

        return self
