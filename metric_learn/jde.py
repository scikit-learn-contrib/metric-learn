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
    def __init__(self, n_gen=25, split_size=0.33, train_subset_size=1.0,
                 max_workers=1, random_state=None, verbose=False):
        """Initialize the learner.

        Parameters
        ----------
        n_gen : int, optional
            Number of generations for the evolution strategy.
        split_size : double, optional
            Ratio of train:test sample size during evolution.
        train_subset_size : double, optional
            Ratio of samples used in training the model during evolution.
        max_workers : int, optional
            Number of workers for parallelization.
        random_state : int, optional
            If provided, controls random number generation.
        verbose : bool, optional
            If true then outputs debugging information.
        """
        super(JDE, self).__init__(
            strategy=st.SelfAdaptingDifferentialEvolution(
                n_gen=n_gen, split_size=split_size, train_subset_size=train_subset_size,
                max_workers=max_workers, random_state=random_state, verbose=verbose
            ),
            fitness_list=[fit.WeightedFMeasureFitness()],
            transformer_func=tr.TriangularMatrixTransformer(),
            random_state=random_state,
            verbose=verbose
        )
