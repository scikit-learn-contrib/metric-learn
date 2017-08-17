class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = None
