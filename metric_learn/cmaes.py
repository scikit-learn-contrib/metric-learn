"""
Using CMA-ES for evolving a Mahalanobis matrix

Based on my Master thesis "Evolutionary Algorithms for Data Transformation"

The thesis can be found at
https://github.com/svecon/thesis-distance-metric-learning/releases/download/1.0/MasterThesis-SvecOndrej.pdf
"""

from sklearn.neighbors import KNeighborsClassifier

from .evolution import MetricEvolution
from .evolution import fitness as fit
from .evolution import strategy as st
from .evolution import transformer as tr


class CMAES(MetricEvolution):
    """
    Using CMA-ES for evolving a Mahalanobis matrix.
    """
    def __init__(self, num_dims=None, **kwargs):
        """Initialize the learner.

        Parameters
        ----------
        num_dims : int, optional
            Dimension of the target space (defaults to the original dimension)
        """
        params = kwargs
        params.update({
            'strategy': st.CMAESEvolution(),
            'fitness_list': [fit.ClassifierFitness(KNeighborsClassifier())],
            'transformer_func': tr.FullMatrixTransformer(num_dims=num_dims),
        })

        super(CMAES, self).__init__(**params)


class CMAES_Diagonal(MetricEvolution):
    """
    Using CMA-ES for evolving a Mahalanobis matrix restricted to a digonal.
    """
    def __init__(self, **kwargs):
        """Initialize the learner."""
        params = kwargs
        params.update({
            'strategy': st.CMAESEvolution(),
            'fitness_list': [fit.ClassifierFitness(KNeighborsClassifier())],
            'transformer_func': tr.DiagonalMatrixTransformer(),
        })

        super(CMAES_Diagonal, self).__init__(**params)
