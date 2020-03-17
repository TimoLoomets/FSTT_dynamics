from keras.models import Sequential
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import Adam

import tensorflow as tf
import numpy as np
import cv2
import yaml
from tqdm import tqdm

from collections import deque
import random
import time
from math import sin, cos, pi
from os import path, getcwd, makedirs

from cone_filter_sorter_node import ConeFilterNode
from interpolator import Interpolator

DISCOUNT = 0.99

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MODEL_NAME = "RELUx3"
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MIN_REWARD = -200

epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

EPISODES = 100
AGGREGATE_STATS_EVERY = 10  # episodes
SHOW_PREVIEW = True


class Car:
    def __init__(self, x, y, phi):
        self._x = x
        self._y = y
        self._phi = phi

        self._width = 0.8
        self._height = 1.5
        self.width = FSEnv.pixels_per_meter * self._width
        self.height = FSEnv.pixels_per_meter * self._height
        self.speed = 0
        self.omega = 0
        self.speed_max = 32
        self.a_max = 7.7

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def phi(self):
        return self._phi

    @property
    def area(self):
        return self._width * self._height

    @property
    def location(self):
        return self._x / FSEnv.pixels_per_meter, self._y / FSEnv.pixels_per_meter

    @property
    def angular_speed_value(self):
        return self.omega

    @property
    def linear_speed_value(self):
        return self.speed

    def rotate_point(self, x, y):
        x_ = x * cos(self.phi) - y * sin(self.phi)
        y_ = x * sin(self.phi) + y * cos(self.phi)
        return x_, y_

    def get_corners(self):
        points = [
            self.rotate_point(-self._width / 2, -self._height / 2),
            self.rotate_point(-self._width / 2, self._height / 2),
            self.rotate_point(self._width / 2, self._height / 2),
            self.rotate_point(self._width / 2, -self._height / 2)
        ]
        for i in range(len(points)):
            points[i] = (
                points[i][0] + self._x / FSEnv.pixels_per_meter, points[i][1] + self._y / FSEnv.pixels_per_meter)
        return points

    def get_points(self):
        points = [
            self.rotate_point(-self.width / 2, -self.height / 2),
            self.rotate_point(-self.width / 2, self.height / 2),
            self.rotate_point(self.width / 2, self.height / 2),
            self.rotate_point(self.width / 2, -self.height / 2)
        ]
        for i in range(len(points)):
            points[i] = (points[i][0] + self._x + FSEnv.offset_x, points[i][1] + self._y + FSEnv.offset_y)
        return points

    def move(self, distance):
        movement = self.rotate_point(0, distance)
        self._x += movement[0]
        self._y += movement[1]

    def rotate(self, angle):
        self._phi += angle

    def update(self, t, control):
        # control_values = control.item(0)
        self.accelerate(control[0])
        self.steer(control[1])

        half_angle = self.omega / 2 * t
        self.rotate(half_angle)
        self.move(self.speed * t * FSEnv.pixels_per_meter)
        self.rotate(half_angle)

        self.speed *= 0.99
        self.omega *= 0.99

    def accelerate(self, throttle):
        self.speed += min(max(throttle, -1), 1) * self.a_max * (1 - self.speed / self.speed_max)
        self.speed = min(max(self.speed, 0), self.speed_max)

    def steer(self, steering):
        self.omega = min(max(steering, -1), 1) * 0.436332 * 5


class FSEnv:
    pixels_per_meter = 8
    offset_x = 30 * pixels_per_meter
    offset_y = 80 * pixels_per_meter

    def __init__(self):
        self.STEP_PENALTY = 1
        self.CHECKPOINT_REWARD = 50
        self.OOB_PENALTY = 100
        self.episode_step = 0
        self.car = None
        self.OBSERVATION_SPACE_VALUES = (6, 2)  # Linear speed and angular speed and then 5 Point Pairs
        self.TIME_STEP = 0.1
        self.ACTION_SPACE_SIZE = 10 * 3#(10, 3)  # throttle, steering, quality triplets # 20 * 20  # throttle and steering

        self.size = 100
        self.img = np.zeros((700, 700, 3), np.uint8)
        self.track = {}
        self.time_factor = 1
        self.sorter = ConeFilterNode()
        self.center_points = []
        self.checkpoints = deque()

        self.load_track()
        self.car = Car(self.track["starting_pose_front_wing"][0], self.track["starting_pose_front_wing"][1],
                       self.track["starting_pose_front_wing"][2] + pi / 2)
        self.calculate_center_line()

        self.reset()

    def clear(self):
        self.img = np.zeros((700, 700, 3), np.uint8)

    @staticmethod
    def point_to_coord(point):
        return round(FSEnv.pixels_per_meter * point[0]) + FSEnv.offset_x, round(
            FSEnv.pixels_per_meter * point[1]) + FSEnv.offset_y

    def draw_cone(self, cone, color):
        center = cone
        cv2.circle(self.img, center, 2, color, -1)

    def draw_car(self):
        pts = np.array(self.car.get_points(), np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.img, [pts], True, (0, 0, 255))

    def check_track(self):
        corners = self.car.get_corners()
        area = self.car.area
        for cone in self.track["cones_left"]:
            if self.point_in_rectangle(cone, corners, area):
                return True
        for cone in self.track["cones_right"]:
            if self.point_in_rectangle(cone, corners, area):
                return True
        return False

    def check_checkpoints(self):
        corners = self.car.get_corners()
        area = self.car.area
        for i in range(2):
            if self.point_in_rectangle(self.checkpoints[i], corners, area):
                for j in range(i + 1):
                    self.checkpoints.popleft()
                break

    @staticmethod
    def triangle_area(a, b, c):
        return abs((b[0] * a[1] - a[0] * b[1]) + (c[0] * b[1] - b[0] * c[1]) + (a[0] * c[1] - c[0] * a[1])) / 2

    @staticmethod
    def point_in_rectangle(point, rectangle, rectangle_area):
        total_area = 0
        for i in range(len(rectangle)):
            total_area += FSEnv.triangle_area(point, rectangle[i], rectangle[i - 1])
        return abs(total_area - rectangle_area) < 0.001

    def load_track(self):
        self.track = yaml.load(open(path.join('FSG.yaml'), 'r'), Loader=yaml.FullLoader)
        print(self.track)

    def calculate_center_line(self):
        self.sorter.pose_update(self.car)
        sorted_pairs = self.sorter.map_update((self.track["cones_right"], self.track["cones_left"]))
        print("yellows:", sorted_pairs.yellowCones)
        print("blues:", sorted_pairs.blueCones)
        # print(len(sorted_pairs.yellowCones), len(sorted_pairs.blueCones))
        for i in range(len(sorted_pairs.yellowCones)):
            left_cone = sorted_pairs.blueCones[i]
            right_cone = sorted_pairs.yellowCones[i]
            self.center_points.append(((left_cone.x + right_cone.x) / 2, (left_cone.y + right_cone.y) / 2))

    def draw_track(self):
        last_cone = None
        for cone in self.track["cones_left"]:
            cone = FSEnv.point_to_coord(cone)
            self.draw_cone(cone, (255, 0, 0))
            if last_cone is not None:
                cv2.line(self.img, last_cone, cone, (255, 0, 0), 1)
            last_cone = cone
        last_cone = None
        for cone in self.track["cones_right"]:
            cone = FSEnv.point_to_coord(cone)
            self.draw_cone(cone, (0, 255, 255))
            if last_cone is not None:
                cv2.line(self.img, last_cone, cone, (0, 255, 255), 1)
            last_cone = cone

        last_cone = None
        # col = 255
        for cone in self.checkpoints:
            cone = FSEnv.point_to_coord(cone)
            self.draw_cone(cone, (0, 255, 0))
            # col = round(0.98 * col)
            if last_cone is not None:
                cv2.line(self.img, last_cone, cone, (0, 255, 0), 1)
            last_cone = cone

    def reset(self):
        self.car = Car(self.track["starting_pose_front_wing"][0], self.track["starting_pose_front_wing"][1],
                       self.track["starting_pose_front_wing"][2] - pi / 2)
        self.checkpoints = deque(self.center_points)
        return self.get_observations()

    def get_observations(self):
        observations = list(self.checkpoints)[:5]
        output = []
        for point in observations:
            loc = self.car.location
            phi = self.car.phi
            x = point[0] - loc[0]
            y = point[1] - loc[1]
            x_ = x * cos(phi) - y * sin(phi)
            y_ = x * sin(phi) + y * cos(phi)
            output.append((x_, y_))

        dt = np.dtype('float')
        return np.array(output, dtype=dt)

    def render(self):
        self.clear()
        self.draw_track()
        self.draw_car()
        cv2.imshow('image', self.img)
        cv2.waitKey(40)  # int(1000 * self.TIME_STEP))

    def step(self, action):
        self.episode_step += 1
        throttle = action[0]  # (action % 20 - 10) / 10.0
        steering = action[1]  # (action // 20 - 10) / 10.0
        self.car.update(self.TIME_STEP, (throttle, steering))

        if self.check_track():
            reward = -self.OOB_PENALTY
            self.reset()
        elif self.check_checkpoints():
            reward = self.CHECKPOINT_REWARD
        else:
            reward = -self.STEP_PENALTY

        new_observation = self.get_observations()

        done = False
        if reward == -self.OOB_PENALTY or self.episode_step >= 6000:
            done = True

        return new_observation, reward, done


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.step = 1
        self._log_write_dir = getcwd()

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        pass
        # self._write_logs(stats, self.step)


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

    @staticmethod
    def create_model():
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=env.OBSERVATION_SPACE_VALUES))
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

        model.add(Dense(env.ACTION_SPACE_SIZE, activation='softmax'))  # ACTION_SPACE_SIZE = how many choices (9)
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]

    # Trains main network every step during episode
    def train(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)

        x = []
        y = []
        interpolator = Interpolator()

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index][:, 2])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_actions = current_qs[:, :2]
            current_qualities = current_qs[:, 2]

            interpolator.set_u(current_actions)
            interpolator.set_q(current_qualities)
            interpolator.update_function(action, new_q)
            current_qs = np.zeros((10, 3))
            current_qs[:, :2] = interpolator.get_u()
            current_qs[:, 2] = interpolator.get_q()

            # print(current_state)
            # print(current_qs_list)
            # print(action)
            # current_qs[action] = new_q

            # And append to our training data
            x.append(current_state)
            y.append(current_qs)

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


if __name__ == "__main__":
    start_time = time.time()
    env = FSEnv()

    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    if not path.isdir('models'):
        makedirs('models')

    agent = DQNAgent()

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            if np.random.random() > epsilon:
                # Get action from Q table
                action_qualities = agent.get_qs(current_state)
                qualities = action_qualities[:, 2]
                best_index = np.argmax(qualities)
                action = action_qualities[best_index][:2]
            else:
                # Get random action
                action = np.array([random.random() * 2 - 1,
                                   random.random() * 2 - 1])  # np.random.randint(0, env.ACTION_SPACE_SIZE)  # np.array(act, dtype=dt)

            new_state, reward, done = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            #    env.render()

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)
            current_state = new_state
            step += 1

        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            print("rewards: ", min_reward, average_reward, max_reward)
            if average_reward >= MIN_REWARD:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    end_time = time.time()
    print("total time:", end_time - start_time)
