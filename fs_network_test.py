import numpy as np

from keras.models import load_model
from fs_network import FSEnv
from output_visualizer import OutputVisualizer
from input_visualizer import InputVisualizer
from constants import *

from collections import deque

model = load_model("models/RELUx3__score___26.29.model")
model.summary()
for layer in model.layers:
    print(layer.input_shape, "=>", layer.output_shape)

env = FSEnv()


###
'''
def assert_close(a, b):
    assert abs(a[0] - b[0]) < 0.001
    assert abs(a[1] - b[1]) < 0.001


env.checkpoints = deque()
env.checkpoints.append((-2.5, 2))
state = env.get_observations()
assert_close(state[1], (-2.5, 2))
print("TEST 1 PASSED")

env.checkpoints = deque()
env.checkpoints.append((2.5, 2))
state = env.get_observations()
assert_close(state[1], (2.5, 2))
print("TEST 2 PASSED")

env.checkpoints = deque()
env.checkpoints.append((2.5, -2))
state = env.get_observations()
assert_close(state[1], (2.5, -2))
print("TEST 3 PASSED")

env.checkpoints = deque()
env.checkpoints.append((-2.5, -2))
state = env.get_observations()
assert_close(state[1], (-2.5, -2))
print("TEST 4 PASSED")
'''
###

output_visualizer = OutputVisualizer()
input_visualizer = InputVisualizer()

while True:
    state = env.get_observations()
    input_visualizer.render(state)
    # action = np.argmax(model.predict(np.array(state).reshape(-1, *state.shape)))
    action_qualities = model.predict(np.array([state]))
    # output_visualizer.render(action_qualities)
    qualities = action_qualities  # [:, 2]
    best_index = np.argmax(qualities)
    action = ACTIONS[best_index]  # [:2]
    print(action)

    env.step(action, True)

    # env.render()
