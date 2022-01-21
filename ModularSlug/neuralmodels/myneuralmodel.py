import numpy as np
from .genericneuralmodel import GenericNeuralModel


class MyNeuralModel(GenericNeuralModel):

    def step(self):
        self.t += 1

        # do some calculations using self.params and self.x[self.t-1]

        # store this iteration's outputs in x
        # - just making up numbers up here
        self.x[self.t]['B3'] = 3 * self.t
        self.x[self.t]['B8'] = 8 * self.t
        self.x[self.t]['B31'] = 31 * self.t
        self.x[self.t]['B38'] = np.nan  # B38 not implemented in this model
