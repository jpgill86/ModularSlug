import numpy as np
from .genericmusclemodel import GenericMuscleModel


class MyMuscleModel(GenericMuscleModel):

    def step(self):

        # do some calculations using self.params, self.neural_outputs, and self.muscle_outputs

        # store this iteration's outputs in x
        # - just making up numbers up here
        self.x[self.t + 1]['I2'] = 0.5 * self.muscle_outputs['I2'] + self.neural_outputs['B31']
        self.x[self.t + 1]['I3'] = 0.5 * self.muscle_outputs['I3'] + self.neural_outputs['B3']
        self.x[self.t + 1]['I4'] = 0.5 * self.muscle_outputs['I4'] + self.neural_outputs['B8']

        self.t += 1
