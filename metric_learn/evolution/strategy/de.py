from deap import tools

import numpy as np

from .base import BaseEvolutionStrategy


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
        toolbox.register(
            "individual", self.generate_individual_with_fitness,
            toolbox.attr_float, individual_size)
        toolbox.register(
            "population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", tools.selRandom, k=3)

        toolbox.register("evaluate", self.evaluation_builder(X, y))

        self.hall_of_fame = tools.HallOfFame(1)
        stats = self._build_stats()

        CR = self.params['cr']
        F = self.params['f']
        pop = toolbox.population(n=self.params['population_size'])

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
            if self.params['verbose']:
                print(self.logbook.stream)

        for g in range(1, self.params['n_gen']):
            for k, agent in enumerate(pop):
                a, b, c = toolbox.select(pop)
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
                if self.params['verbose']:
                    print(self.logbook.stream)

        return self
