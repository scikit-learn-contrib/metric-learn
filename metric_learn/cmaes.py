# https://github.com/svecon/thesis-distance-metric-learning/releases/tag/1.0
from sklearn.neighbors import KNeighborsClassifier

from .evolution import MetricEvolution
from .evolution import fitness as fit
from .evolution import strategy as st
from .evolution import transformer as tr


class CMAES(MetricEvolution):
    def __init__(self, num_dims=None, **kwargs):
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
        params = kwargs
        params.update({
            'strategy': st.CMAESEvolution(),
            'fitness': [fit.ClassifierFitness(KNeighborsClassifier())],
            'transformer_func': tr.FullMatrixTransformer(num_dims=num_dims),
        })

        super(CMAES, self).__init__(**params)


class CMAES_Diagonal(MetricEvolution):
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
        params = kwargs
        params.update({
            'strategy': st.CMAESEvolution(),
            'fitness': [fit.ClassifierFitness(KNeighborsClassifier())],
            'transformer_func': tr.DiagonalMatrixTransformer(),
        })

        super(CMAES_Diagonal, self).__init__(**params)
