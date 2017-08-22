# https://github.com/svecon/thesis-distance-metric-learning/releases/tag/1.0

from .evolution import MetricEvolution
from .evolution import fitness as fit
from .evolution import strategy as st
from .evolution import transformer as tr


class JDE(MetricEvolution):
    def __init__(self, **kwargs):
        """Initialize the learner.

        Parameters
        ----------
        TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!
        transformer_shape : ('full', 'diagonal', MatrixTransformer object)
            transformer_func shape defines transforming function to learn
        num_dims : int, optional
            Dimensionality of reduced space (defaults to dimension of X)
        verbose : bool, optional
            if True, prints information while learning
        """
        super(JDE, self).__init__(
            strategy=st.SelfAdaptingDifferentialEvolution(),
            fitness=[fit.WeightedFMeasureFitness()],
            transformer_func=tr.TriangularMatrixTransformer(),
            **kwargs,
        )
