import numpy as np


muscle_outputs_dtype = [
    ('I2', 'float64'),
    ('I3', 'float64'),
    ('I4', 'float64'),
]


class GenericMuscleModel:

    def __init__(self, params, n_steps, x0):
        if x0.dtype != muscle_outputs_dtype:
            raise TypeError(f'initial conditions x0 do not have the right data type for {self.__class__.__name__}: {x0.dtype}')

        self.params = params
        self.x = np.zeros(n_steps+1, dtype=muscle_outputs_dtype)
        self.x[0] = x0
        self.t = 0

    def step(self):
        raise NotImplementedError
