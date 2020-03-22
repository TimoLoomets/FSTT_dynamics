import numpy as np

from keras.models import load_model
from fs_network import FSEnv, DQNAgent

model = load_model("models/RELUx3___403.00max__139.90avg___25.00min__1584906951.model")
model.summary()
for layer in model.layers:
    print(layer.input_shape, "=>", layer.output_shape)

env = FSEnv()

while True:
    state = env.get_observations()
    # action = np.argmax(model.predict(np.array(state).reshape(-1, *state.shape)))
    action_qualities = model.predict(np.array([state]))
    qualities = action_qualities[:, 2]
    best_index = np.argmax(qualities)
    action = action_qualities[best_index][:2]
    print(action)

    env.step(action)

    #env.render()
