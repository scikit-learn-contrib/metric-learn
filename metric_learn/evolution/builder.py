# The CMA-ES algorithm takes a population of one individual as argument
# See http://www.lri.fr/~hansen/cmaes_inmatlab.html
# for more details about the rastrigin and other tests for CMA-ES

from . import fitness as fit
from . import strategy as st
from . import transformer as tr
from .evolution import MetricEvolution


class MetricEvolutionBuilder(MetricEvolution):
    def __init__(self, strategy, fitness, transformer_func, **kwargs):
        """Initialize the learner.

        Parameters
        ----------
        fitness : ('knn', 'svc', 'lsvc', fitness object)
            fitness is used in fitness scoring
        transformer_func : ('full', 'diagonal', MatrixTransformer object)
            transformer_func shape defines transforming function to learn
        num_dims : int, optional
            Dimensionality of reduced space (defaults to dimension of X)
        verbose : bool, optional
            if True, prints information while learning
        """
        super(MetricEvolutionBuilder, self).__init__(
            strategy=self.build_strategy(strategy),
            fitness=self.build_fitnesses(fitness),
            transformer_func=self.build_transformer(transformer_func),
            **kwargs,
        )

    def build_fitnesses(self, fitnesses):
        # make it an array if it is not already one
        if not isinstance(fitnesses, (list, tuple)):
            fitnesses = [fitnesses]

        return [self.build_fitness(f) for f in fitnesses]

    def build_fitness(self, fitness):
        # fitness can be a tuple of fitness and its params
        params = dict()
        if isinstance(fitness, (list, tuple)):
            fitness, params = fitness

        if isinstance(fitness, fit.BaseFitness):
            return fitness
        elif fit.ClassifierFitness.available(fitness):
            return fit.ClassifierFitness(fitness, **params)
        elif fit.RandomFitness.available(fitness):
            return fit.RandomFitness(**params)
        elif fit.ClassSeparationFitness.available(fitness):
            return fit.ClassSeparationFitness(**params)
        elif fit.WeightedPurityFitness.available(fitness):
            return fit.WeightedPurityFitness(**params)
        elif fit.WeightedFMeasureFitness.available(fitness):
            return fit.WeightedFMeasureFitness(**params)

        raise ValueError('Invalid `fitness` value: `{}`'.format(fitness))

    def build_strategy(self, strategy):
        if isinstance(strategy, st.BaseEvolutionStrategy):
            return strategy
        elif strategy == 'cmaes':
            return st.CMAESEvolution()
        elif strategy == 'de':
            return st.DifferentialEvolution()
        elif strategy == 'jde':
            return st.SelfAdaptingDifferentialEvolution()
        elif strategy == 'dde':
            return st.DynamicDifferentialEvolution()

        raise ValueError('Invalid `strategy` value: `{}`'.format(strategy))

    def build_transformer(self, transformer):
        if isinstance(transformer, tr.MatrixTransformer):
            return transformer
        elif transformer == 'diagonal':
            return tr.DiagonalMatrixTransformer()
        elif transformer == 'full':
            return tr.FullMatrixTransformer()
        elif transformer == 'triangular':
            return tr.TriangularMatrixTransformer()
        elif transformer == 'neuralnetwork':
            return tr.NeuralNetworkTransformer()
        elif transformer == 'kmeans':
            return tr.KMeansTransformer()

        raise ValueError(
            'Invalid `transformer` value: `{}`'.format(transformer))
