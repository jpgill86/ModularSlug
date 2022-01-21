import numpy as np


class Aplysia:

    def __init__(self, n_steps, neural_model, muscle_model):
        self.n_steps = n_steps
        self.neural_model = neural_model
        self.muscle_model = muscle_model

    def run(self):
        for i in range(self.n_steps):
            self.neural_model.step()
            self.muscle_model.step()

    def summarize(self):
        print('-- Neural Model --')
        print('params:', self.neural_model.params)
        print('x:', self.neural_model.x)

        print()

        print('-- Muscle Model --')
        print('params:', self.muscle_model.params)
        print('x:', self.muscle_model.x)
