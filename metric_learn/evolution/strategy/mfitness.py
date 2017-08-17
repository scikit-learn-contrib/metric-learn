from deap import base


class MultidimensionalFitness(base.Fitness):
    def __init__(self, n_dim, *args, **kwargs):
        self.n_dim = n_dim
        self.weights = (1.0,) * n_dim

        super().__init__(*args, **kwargs)

    def __deepcopy__(self, memo):
        copy_ = self.__class__(self.n_dim)
        copy_.wvalues = self.wvalues
        return copy_
