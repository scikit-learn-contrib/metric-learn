from deap import tools

import numpy as np

from .base_strategy import BaseEvolutionStrategy


class DifferentialEvolution(BaseEvolutionStrategy):
    def __init__(self, population_size=50, cr=.25, f=1.0, **kwargs):
        super().__init__(**kwargs)

        self.population_size = population_size
        self.cr = cr
        self.f = f

    def best_individual(self):
        return self.hall_of_fame[0]

    def fit(self, X, y):
        individual_size = self.n_dim

        toolbox = self.create_toolbox()
        toolbox.register("attr_float", np.random.uniform, -1, 1)
        toolbox.register(
            "individual", self.generate_individual_with_fitness,
            toolbox.attr_float, individual_size)
        toolbox.register(
            "population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", self.evaluation_builder(X, y))

        self.hall_of_fame = tools.HallOfFame(1)
        stats = self._build_stats()

        pop = toolbox.population(n=self.population_size)

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
                index = np.random.randint(individual_size)
                for i, value in enumerate(agent):
                    if i == index or np.random.random() < self.cr:
                        y[i] = a[i] + self.f * (b[i] - c[i])
                y.fitness.values = toolbox.evaluate(y)
                if y.fitness > agent.fitness:
                    pop[k] = y
            self.hall_of_fame.update(pop)

            if stats:
                record = stats.compile(pop)
                self.logbook.record(gen=g, evals=len(pop), **record)
                if self.verbose:
                    print(self.logbook.stream)

        return self
