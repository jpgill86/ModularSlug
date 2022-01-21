import numpy as np


neural_outputs_dtype = [
    ('B3', 'float64'),
    ('B8', 'float64'),
    ('B31', 'float64'),
    ('B38', 'float64'),
]


class GenericNeuralModel:

    def __init__(self, params, n_steps, x0):

        if not isinstance(x0, np.ndarray) or x0.dtype != neural_outputs_dtype:
            try:
                # attempt to convert x0 to a structured array
                x0 = np.array(tuple(x0), dtype=neural_outputs_dtype)
            except:
                raise TypeError('initial conditions x0 does not have the right '
                                f'data type for {self.__class__.__name__} and '
                                'could not be converted automatically')

        self.params = params
        self.x = np.zeros(n_steps+1, dtype=neural_outputs_dtype)
        self.x[0] = x0
        self.t = 0

    def step(self):
        raise NotImplementedError
