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
    def __init__(self, num_dims=None, n_gen=25, split_size=0.33,
                 train_subset_size=1.0, max_workers=1, random_state=None,
                 verbose=False):
        """Initialize the learner.

        Parameters
        ----------
        num_dims : int, optional
            Dimension of the target space (defaults to the original dimension).
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
        super(CMAES, self).__init__(
            strategy=st.CMAESEvolution(
                n_gen=n_gen, split_size=split_size, train_subset_size=train_subset_size,
                max_workers=max_workers, random_state=random_state, verbose=verbose
            ),
            fitness_list=[fit.ClassifierFitness(KNeighborsClassifier())],
            transformer_func=tr.FullMatrixTransformer(num_dims=num_dims),
            random_state=random_state,
            verbose=verbose
        )


class CMAES_Diagonal(MetricEvolution):
    """
    Using CMA-ES for evolving a Mahalanobis matrix restricted to a digonal.
    """
    def __init__(self, n_gen=25, split_size=0.33, train_subset_size=1.0,
                 max_workers=1, random_state=None, verbose=False):
        """Initialize the learner.
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
        super(CMAES_Diagonal, self).__init__(
            strategy=st.CMAESEvolution(
                n_gen=n_gen, split_size=split_size, train_subset_size=train_subset_size,
                max_workers=max_workers, random_state=random_state, verbose=verbose
            ),
            fitness_list=[fit.ClassifierFitness(KNeighborsClassifier())],
            transformer_func=tr.DiagonalMatrixTransformer(),
            random_state=random_state,
            verbose=verbose
        )
