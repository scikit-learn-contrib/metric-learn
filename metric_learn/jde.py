"""
Using jDE for evolving a triangular Mahalanobis matrix
based on Fukui 2013: Evolutionary Distance Metric Learning Approach
to Semi-supervised Clustering with Neighbor Relations

There are some notable differences between the paper and this implementation,
please refer to
https://github.com/svecon/thesis-distance-metric-learning/releases/tag/1.0
"""

from .evolution import MetricEvolution
from .evolution import fitness as fit
from .evolution import strategy as st
from .evolution import transformer as tr


class JDE(MetricEvolution):
    """
    Using jDE for evolving a triangular Mahalanobis matrix.
    """
    def __init__(self, **kwargs):
        """Initialize the learner."""
        params = kwargs
        params.update({
            'strategy': st.SelfAdaptingDifferentialEvolution(),
            'fitness_list': [fit.WeightedFMeasureFitness()],
            'transformer_func': tr.TriangularMatrixTransformer(),
        })
        super(JDE, self).__init__(**params)
