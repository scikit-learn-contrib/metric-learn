"""
MetricEvolutionBuilder is an extension of MetricEvolution whose purpose
is to make it easier to instantiate the class.

Therefore instead of instantiating all the parameters, one can just pass
string representations of the strategy, fitness and transformer.
"""

from . import fitness as fit
from . import strategy as st
from . import transformer as tr
from .evolution import MetricEvolution


class MetricEvolutionBuilder(MetricEvolution):
    def __init__(self, strategy, fitness, transformer_func, **kwargs):
        """Initialize the evolutionary learner.

        Parameters
        ----------
        strategy : (str, BaseEvolutionStrategy object)
            evolution strategy used for the optimization.
                'de' - basic Differential Evolution
                'dde' - Dynamic Differential Evolution
                'sade' - Self-adapting Differential Evolution
                'cmaes' - CMA-ES
        fitness : (str, BaseFitness object)
            fitness is used in fitness scoring
                'knn' - kNearestNeighbour classifier (by ClassifierFitness)
                'svc' - SVM classifier (by ClassifierFitness)
                'lsvc' - linear SVM classifier (by ClassifierFitness)
                'wfme' - weighted Fmeasure score
                'wpur' - weighted purity score
                'random' - random fitness score
                'class_separation' - class separation score
        transformer_func : ('str', BaseTransformer object)
            transformer_func defines transforming function to be learnt
                'full' - full Mahalanobis matrix
                'diagonal' - matrix restricted to diagonal
                'triangular' - upper triangular matrix
                'neuralnetwork' - fully connected neural network
        """
        params = kwargs
        params.update({
            'strategy': self.build_strategy(strategy),
            'fitness_list': self.build_fitnesses(fitness),
            'transformer_func': self.build_transformer(transformer_func),
        })
        super(MetricEvolutionBuilder, self).__init__(**params)

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

        raise ValueError(
            'Invalid `transformer` value: `{}`'.format(transformer))
