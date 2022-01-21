import numpy as np


class GenericNeuralModel:

    neural_outputs_dtype = [
        ('B3', 'float64'),
        ('B8', 'float64'),
        ('B31', 'float64'),
        ('B38', 'float64'),
    ]

    def __init__(self, params, n_steps, x0):
        self._parent = None

        if not isinstance(x0, np.ndarray) or x0.dtype != self.neural_outputs_dtype:
            try:
                # attempt to convert x0 to a structured array
                x0 = np.array(tuple(x0), dtype=self.neural_outputs_dtype)
            except:
                raise TypeError('initial conditions x0 does not have the right '
                                f'data type for {self.__class__.__name__} and '
                                'could not be converted automatically')

        self.params = params
        self.x = np.zeros(n_steps+1, dtype=self.neural_outputs_dtype)
        self.x[0] = x0
        self.t = 0

    @property
    def parent(self):
        # defining this property is required for creating its custom setter
        return self._parent

    @parent.setter
    def parent(self, obj):
        # ensure parent can only be set to an instance of Aplysia or to None
        from ..aplysia import Aplysia
        if not isinstance(obj, (Aplysia, type(None))):
            raise TypeError('tried to set parent to an incompatible '
                            f'object type: {obj.__class__.__name__}')

        self._parent = obj

    @property
    def outputs(self):
        '''The current values of the outputs'''
        return self.x[self.t]

    def step(self):
        raise NotImplementedError
