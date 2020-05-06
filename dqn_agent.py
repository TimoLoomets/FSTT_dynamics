from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam
import time
import random
import numpy as np

from constants import *
from modified_tensor_board import ModifiedTensorBoard
from interpolator import Interpolator
from output_visualizer import OutputVisualizer

class DQNAgent:
    def __init__(self):
        # Main model
        self.model = self.create_model()

        # Target network
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Custom tensorboard object
        self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))

        # Used to count when to update target network with main network's weights
        self.target_update_counter = 0

        # Visualization
        self.output_visualizer = OutputVisualizer()

    @staticmethod
    def create_model():
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=INPUT_2D_SHAPE))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))

        '''
        model.add(Conv1D(256, 2,
                         input_shape=env.OBSERVATION_SPACE_VALUES))  # OBSERVATION_SPACE_VALUES = (10, 10, 3) a 10x10 RGB image.
        model.add(Activation('relu'))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Conv1D(256, 2))
        model.add(Activation('relu'))
        # model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        '''

        model.add(Dense(OUTPUT_1D_SHAPE, activation='linear'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])

        for layer in model.layers:
            print(layer.output_shape)
        model.summary()

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        output = np.reshape(self.model.predict(np.array(state).reshape(-1, *state.shape))[0], OUTPUT_2D_SHAPE)
        self.output_visualizer.render(np.concatenate((np.array(ACTIONS), output), axis=1))
        return output

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])# / 255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])# / 255
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []
        interpolator = Interpolator()

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            future_qs = np.reshape(future_qs_list[index], OUTPUT_2D_SHAPE)
            if not done:
                max_future_q = np.max(future_qs)
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = np.reshape(current_qs_list[index], OUTPUT_2D_SHAPE)
            current_actions = ACTIONS
            current_qualities = current_qs

            interpolator.set_u(current_actions)
            interpolator.set_q(current_qualities)
            interpolator.update_function(action, new_q)
            #current_qs = np.zeros(OUTPUT_2D_SHAPE)
            #current_qs[:, :2] = interpolator.get_u()
            current_qs = interpolator.get_q()

            # print(current_state)
            # print(current_qs_list)
            # print(action)
            # current_qs[action] = new_q

            # And append to our training data
            x.append(current_state)
            y.append(np.reshape(current_qs, OUTPUT_1D_SHAPE))

        # print("x:", x)
        # print("y:", y)
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(x) / 255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
