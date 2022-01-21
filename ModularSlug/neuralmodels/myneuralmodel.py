import numpy as np
from .genericneuralmodel import GenericNeuralModel


class MyNeuralModel(GenericNeuralModel):

    def step(self):

        # do some calculations using self.params, self.neural_outputs, and self.muscle_outputs

        # store this iteration's outputs in x
        # - just making up numbers up here
        self.x[self.t + 1]['B3'] = 3 * (self.t + 1)
        self.x[self.t + 1]['B8'] = 8 * (self.t + 1)
        self.x[self.t + 1]['B31'] = 31 * (self.t + 1)
        self.x[self.t + 1]['B38'] = np.nan  # B38 not implemented in this model

        self.t += 1
