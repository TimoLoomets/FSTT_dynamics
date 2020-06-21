import numpy as np

from keras.models import load_model
from fs_network import FSEnv
from output_visualizer import OutputVisualizer
from input_visualizer import InputVisualizer
from constants import *

from collections import deque

state_model = load_model("models/1592767960/__trained_state.model")
state_model.summary()
for layer in state_model.layers:
    print(layer.input_shape, "=>", layer.output_shape)

quality_model = load_model("models/1592767960/__trained_quality.model")
quality_model.summary()
for layer in quality_model.layers:
    print(layer.input_shape, "=>", layer.output_shape)

env = FSEnv()

output_visualizer = OutputVisualizer()
input_visualizer = InputVisualizer()

CALCULATION_DEPTH = 3
DISCOUNT = 0.95


def get_qualities(state, depth=0):
    if depth >= CALCULATION_DEPTH:
        return [0]

    qualities = []
    for action in ACTIONS:
        prediction_input = np.array([list(state) + [list(action)]])
        state_prediction = state_model.predict(prediction_input)
        reshaped_prediction = state_prediction.reshape(-2, 2)
        next_state = list(map(list, list(reshaped_prediction)))
        r = quality_model.predict(prediction_input)[0][0]
        qualities.append(r + DISCOUNT * max(get_qualities(next_state, depth=depth + 1)))
    return qualities  # [::-1]


# flag=True
for step in range(EPISODE_LENGTH):
    state = env.get_observations()
    input_visualizer.render(state)
    # action = np.argmax(model.predict(np.array(state).reshape(-1, *state.shape)))
    action_qualities = get_qualities(state)  # model.predict(np.array([state]))
    output_visualizer.render(
        np.concatenate((np.array(ACTIONS), np.reshape(np.array(action_qualities), OUTPUT_2D_SHAPE)), axis=1))
    qualities = action_qualities  # [:, 2]
    best_index = np.argmax(qualities)
    action = ACTIONS[best_index]  # tuple(map(lambda x: -1*x, ACTIONS[best_index]))  # [:2]
    print(action)

    _, reward, _ = env.step(action, True)
    print("reward:", reward)
    # if flag:
    #    input()
    #    flag = False
    # env.render()
