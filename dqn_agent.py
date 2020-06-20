from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam, RMSprop
import tensorflow as tf
import time
import random
import numpy as np
import pandas as pd

from constants import *
from modified_tensor_board import ModifiedTensorBoard
from interpolator import Interpolator
from output_visualizer import OutputVisualizer
from input_visualizer import InputVisualizer


class DQNAgent:
    def __init__(self, start_time):
        # Main model
        self.model = self.create_model()
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time
        tf.keras.utils.plot_model(
            self.model,
            to_file='logs/' + TRACK_FILE.split('.')[0] + '/'
                    + str(round(self.start_time)) + "/model.png",
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=False,
            dpi=96,
        )

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
        self.input_visualizer = InputVisualizer()

    @staticmethod
    def create_model():
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=INPUT_2D_SHAPE))
        model.add(Flatten())
        for _ in range(7):
            model.add(Dense(50, activation='relu'))
        # model.add(Dense(125, activation='sigmoid'))
        model.add(Dense(OUTPUT_1D_SHAPE, activation='linear'))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=RMSprop(lr=0.01), metrics=['accuracy']) # huber_loss

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

        for layer in model.layers:
            print(layer.output_shape)
        model.summary()

        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state, visualize=False):
        predicted_qualities = self.model.predict(np.array(state).reshape(-1, *state.shape))
        reshaped_qualities = predicted_qualities[0]
        output = np.reshape(reshaped_qualities, OUTPUT_2D_SHAPE)
        if visualize:
            self.input_visualizer.render(state)
            self.output_visualizer.render(np.concatenate((np.array(ACTIONS), output), axis=1))
        return output

    def save_replay_memory(self):
        df = pd.DataFrame([[d if type(d) != np.ndarray else d.tolist() for d in e] for e in self.replay_memory],
                          columns=['State', 'Action', 'Reward', 'NextState', 'Done'])
        df.to_csv('logs/' + TRACK_FILE.split('.')[0] + '/' + str(round(self.start_time)) + "/data.csv", index=False)

    # Trains main network every step during episode
    def train(self, terminal_state):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Calculate Prioritized Experience Replay weights
        current_states = np.array([transition[0] for transition in self.replay_memory])
        future_states = np.array([transition[3] for transition in self.replay_memory])
        current_qs = self.model.predict(current_states)
        future_qs = self.target_model.predict(future_states)
        p = np.array([abs((reward + DISCOUNT * np.amax(future_qs[index]) if not done else reward)
                          - current_qs[index][ACTIONS.index(action)])
                      for index, (_, action, reward, _, done) in enumerate(self.replay_memory)])
        p = np.interp(p, (p.min(), p.max()), (0, +1))
        p /= np.sum(p)

        # Get a minibatch of random samples from memory replay table
        minibatch = np.array(self.replay_memory)[np.random.choice(len(self.replay_memory),
                                                                  size=MINIBATCH_SIZE,
                                                                  replace=False,
                                                                  p=p)]  # random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])  # / 255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])  # / 255
        future_target_qs_list = self.target_model.predict(new_current_states)
        future_model_qs_list = self.model.predict(new_current_states)

        x = []
        y = []
        interpolator = Interpolator()

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            future_model_qs_at_index = future_model_qs_list[index]
            future_target_qs_at_index = future_target_qs_list[index]
            # future_qs = np.reshape(future_model_qs_at_index, OUTPUT_2D_SHAPE)
            if not done:
                max_future_q = future_target_qs_at_index[np.argmax(future_model_qs_at_index)]
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs_list_at_index = current_qs_list[index]
            current_qs = np.reshape(current_qs_list_at_index, OUTPUT_2D_SHAPE)
            current_actions = ACTIONS
            current_qualities = current_qs

            interpolator.set_u(current_actions)
            interpolator.set_q(current_qualities)
            interpolator.update_function(action, new_q)
            # current_qs = np.zeros(OUTPUT_2D_SHAPE)
            # current_qs[:, :2] = interpolator.get_u()
            current_qs = interpolator.get_q()  # [current_actions.index(action)] = [new_q]  #

            # print(current_state)
            # print(current_qs_list)
            # print(action)
            # current_qs[action] = new_q

            # And append to our training data
            x.append(current_state)
            reshaped_current_qs = np.reshape(current_qs, OUTPUT_1D_SHAPE)
            y.append(reshaped_current_qs)

        # print("x:", x)
        # print("y:", y)
        # Fit on all samples as one batch, log only on terminal state
        self.model.fit(np.array(x), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            # a = self.model.get_weights()
            # print(a)
            self.target_update_counter = 0
            self.save_replay_memory()
