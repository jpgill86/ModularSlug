import numpy as np
import ModularSlug


n_steps = 10

params = []
x0 = [0, 0, 0, 0]  # corresponds to ModularSlug.neural_outputs_dtype
neural_model = ModularSlug.MyNeuralModel(params, n_steps, x0)

params = []
x0 = [0, 0, 0]  # corresponds to ModularSlug.muscle_outputs_dtype
muscle_model = ModularSlug.MyMuscleModel(params, n_steps, x0)

aplysia = ModularSlug.Aplysia(n_steps, neural_model, muscle_model)
aplysia.run()
aplysia.summarize()
