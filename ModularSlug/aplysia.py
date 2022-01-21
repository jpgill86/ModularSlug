import numpy as np
from .neuralmodels import GenericNeuralModel
from .musclemodels import GenericMuscleModel


class Aplysia:

    def __init__(self, n_steps, neural_model, muscle_model):
        self._neural_model = None
        self._muscle_model = None

        self.n_steps = n_steps
        self.neural_model = neural_model
        self.muscle_model = muscle_model

    @property
    def neural_model(self):
        # defining this property is required for creating its custom setter
        return self._neural_model

    @neural_model.setter
    def neural_model(self, obj):
        # ensure neural_model can only be set to an instance of a subclass of GenericNeuralModel or to None
        if not isinstance(obj, (GenericNeuralModel, type(None))):
            raise TypeError('tried to set neural_model to an incompatible '
                            f'object type: {obj.__class__.__name__}')

        # if there is an old neural_model, first unset its parent
        if self._neural_model is not None:
            self._neural_model.parent = None

        self._neural_model = obj
        self._neural_model.parent = self

    @property
    def muscle_model(self):
        # defining this property is required for creating its custom setter
        return self._muscle_model

    @muscle_model.setter
    def muscle_model(self, obj):
        # ensure muscle_model can only be set to an instance of a subclass of GenericMuscleModel or to None
        if not isinstance(obj, (GenericMuscleModel, type(None))):
            raise TypeError('tried to set muscle_model to an incompatible '
                            f'object type: {obj.__class__.__name__}')

        # if there is an old muscle_model, first unset its parent
        if self._muscle_model is not None:
            self._muscle_model.parent = None

        self._muscle_model = obj
        self._muscle_model.parent = self

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
