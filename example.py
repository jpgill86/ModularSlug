import numpy as np
from ModularSlug import Aplysia, MyNeuralModel, MyMuscleModel


n_steps = 10

params = []
x0 = [0, 0, 0, 0]  # corresponds to MyNeuralModel.neural_outputs_dtype
neural_model = MyNeuralModel(params, n_steps, x0)

params = []
x0 = [0, 0, 0]  # corresponds to MyMuscleModel.muscle_outputs_dtype
muscle_model = MyMuscleModel(params, n_steps, x0)

aplysia = Aplysia(n_steps, neural_model, muscle_model)
aplysia.run()
aplysia.summarize()

print()
print('Final value of B3:', aplysia.neural_outputs['B3'])
print('Final value of I2:', aplysia.muscle_outputs['I2'])
