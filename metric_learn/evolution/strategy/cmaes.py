from deap import algorithms, cma, tools

from .base import BaseEvolutionStrategy, Individual, MultidimensionalFitness


class CMAESEvolution(BaseEvolutionStrategy):
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
        toolbox.register(
            "generate",
            self._generate_pop_with_fitness,
            strategy.generate)
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
