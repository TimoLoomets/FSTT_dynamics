import numpy as np

from keras.models import load_model
from fs_network import FSEnv, DQNAgent

model = load_model("models/256x2____-1.00max___-1.00avg___-1.00min__1582635321.model")
model.summary()

env = FSEnv()

while True:
    state = env.get_observations()
    action = np.argmax(model.predict(np.array(state).reshape(-1, *state.shape)))

    env.step(action)

    env.render()
