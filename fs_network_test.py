import numpy as np

from keras.models import load_model
from fs_network import FSEnv, DQNAgent
from output_visualizer import OutputVisualizer
from input_visualizer import InputVisualizer
from constants import *

model = load_model("models/RELUx3__score____6.79.model")
model.summary()
for layer in model.layers:
    print(layer.input_shape, "=>", layer.output_shape)

env = FSEnv()

output_visualizer = OutputVisualizer()
input_visualizer = InputVisualizer()

while True:
    state = env.get_observations()
    input_visualizer.render(state)
    # action = np.argmax(model.predict(np.array(state).reshape(-1, *state.shape)))
    action_qualities = model.predict(np.array([state]))
    #output_visualizer.render(action_qualities)
    qualities = action_qualities  # [:, 2]
    best_index = np.argmax(qualities)
    action = ACTIONS[best_index]  # [:2]
    print(action)

    env.step(action, True)

    # env.render()
