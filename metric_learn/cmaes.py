# https://github.com/svecon/thesis-distance-metric-learning/releases/tag/1.0

from .evolution import MetricEvolution
from .evolution import fitness as fit
from .evolution import strategy as st
from .evolution import transformer as tr


class CMAES(MetricEvolution):
    def __init__(self, transformer='full', num_dims=None, random_state=None,
                 verbose=False):
        """Initialize the learner.

        Parameters
        ----------
        TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!
        transformer_shape : ('full', 'diagonal', MatrixTransformer object)
            transformer shape defines transforming function to learn
        num_dims : int, optional
            Dimensionality of reduced space (defaults to dimension of X)
        verbose : bool, optional
            if True, prints information while learning
        """
        super(CMAES, self).__init__(
            strategy='cmaes',
            fitnesses='knn',
            transformer=transformer,
            random_state=random_state,
            verbose=verbose,
        )
