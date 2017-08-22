class Individual(list):
    def __init__(self, *args):
        super(Individual, self).__init__(*args)
        self.fitness = None
