import numpy as np
from .genericmusclemodel import GenericMuscleModel


class MyMuscleModel(GenericMuscleModel):

    def step(self):
        self.t += 1

        # do some calculations using self.params and self.x[self.t-1]

        # store this iteration's outputs in x
        # - just making up numbers up here
        self.x[self.t]['I2'] = 2 * self.t
        self.x[self.t]['I3'] = 3 * self.t
        self.x[self.t]['I4'] = 4 * self.t
