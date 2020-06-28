import numpy as np
from collections import deque
from math import sin, cos, pi
import yaml
from os import path
from random import random

# from car import Car
from constants import *
from cone_filter_sorter_node import ConeFilterNode
from simple_car import SimpleCar
from simulator_visualizer import SimulatorVisualizer


class FSEnv:

    def __init__(self):
        self.STEP_PENALTY = 1
        self.CHECKPOINT_REWARD = 20
        self.OOB_PENALTY = 10
        self.episode_step = 0
        self.car = None
        self.OBSERVATION_SPACE_VALUES = INPUT_2D_SHAPE  # Linear speed and angular speed and then 5 Point Pairs
        self.TIME_STEP = 0.1
        self.ACTION_SPACE_SIZE = OUTPUT_1D_SHAPE  # throttle, steering, quality triplets

        self.size = 100

        self.track = {}
        self.active_track = {}
        self.time_factor = 1
        self.sorter = ConeFilterNode()
        self.center_points = []
        self.checkpoints = deque()
        self.visualizer = SimulatorVisualizer()
        self.last_speed_was_zero = False
        self.consecutive_successes = 0
        self.track_index = 0

        self.load_track()
        self.mutate_track(MUTATION_RANGE)
        self.car = SimpleCar(self.track["starting_pose_front_wing"][0],
                             self.track["starting_pose_front_wing"][1],
                             self.track["starting_pose_front_wing"][2] + pi / 2)
        self.calculate_center_line()

        self.reset()

    @staticmethod
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            return None
            # raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    @staticmethod
    def point_in_line(point, line):
        return min(line[0][0], line[1][0]) <= point[0] <= max(line[0][0], line[1][0]) \
               and min(line[0][1], line[1][1]) <= point[1] <= max(line[0][1], line[1][1])

    @staticmethod
    def triangle_area(a, b, c):
        return abs((b[0] * a[1] - a[0] * b[1]) + (c[0] * b[1] - b[0] * c[1]) + (a[0] * c[1] - c[0] * a[1])) / 2

    @staticmethod
    def point_in_rectangle(point, rectangle, rectangle_area):
        total_area = 0
        for i in range(len(rectangle)):
            total_area += FSEnv.triangle_area(point, rectangle[i], rectangle[i - 1])
        return abs(total_area - rectangle_area) < 0.001

    @staticmethod
    def point_in_circle(point, center, radius):
        return (point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2 < radius ** 2

    def check_edges(self, cones):
        car_edges = self.car.get_edges()
        # cones = self.track["cones_left"]
        for cone_index in range(1, len(cones)):
            track_edge = (cones[cone_index - 1], cones[cone_index])
            for car_edge in car_edges:
                intersection = FSEnv.line_intersection(car_edge, track_edge)
                if intersection is not None \
                        and FSEnv.point_in_line(intersection, car_edge) \
                        and FSEnv.point_in_line(intersection, track_edge):
                    return True
        return False

    def check_track(self):
        corners = self.car.get_corners()
        area = self.car.area
        for cone in self.active_track["cones_left"]:
            if self.point_in_rectangle(cone, corners, area):
                return True
        for cone in self.active_track["cones_right"]:
            if self.point_in_rectangle(cone, corners, area):
                return True
        return self.check_edges(self.active_track["cones_left"]) \
               or self.check_edges(self.active_track["cones_right"])

    def check_checkpoints(self):
        corners = self.car.get_corners()
        area = self.car.area
        for i in range(len(self.checkpoints)):  # min(2, len(self.checkpoints))):
            if self.point_in_circle(self.checkpoints[i],
                                    (self.car.x, self.car.y),
                                    1):
                self.checkpoints.remove(self.checkpoints[i])
                # for j in range(i + 1):
                #    self.checkpoints.popleft()
                return True
        return False

    def load_track(self):
        self.track = yaml.load(open(path.join('tracks', TRACK_FILES[self.track_index]), 'r'), Loader=yaml.FullLoader)
        self.active_track = self.track.copy()
        print(self.track)

    def next_track(self):
        self.track_index = min(self.track_index + 1, len(TRACK_FILES) - 1)
        self.load_track()
        self.reset()

    def mutate_track(self, move_range):
        self.active_track['cones_left'] = [[cone[0] + move_range * 2 * (random() - 0.5),
                                            cone[1] + move_range * 2 * (random() - 0.5)]
                                           for cone in self.track['cones_left']]
        self.active_track['cones_right'] = [[cone[0] + move_range * 2 * (random() - 0.5),
                                             cone[1] + move_range * 2 * (random() - 0.5)]
                                            for cone in self.track['cones_right']]

    def calculate_center_line(self):
        self.sorter.pose_update(self.car)
        sorted_pairs = self.sorter.map_update((self.track["cones_right"], self.track["cones_left"]))
        # print("yellows:", sorted_pairs.yellowCones)
        # print("blues:", sorted_pairs.blueCones)
        self.center_points = []
        for i in range(len(sorted_pairs.yellowCones)):
            left_cone = sorted_pairs.blueCones[i]
            right_cone = sorted_pairs.yellowCones[i]
            self.center_points.append(((left_cone.x + right_cone.x) / 2, (left_cone.y + right_cone.y) / 2))
        if len(self.center_points) > 25:
            self.center_points.pop(0)
            self.center_points.pop(0)

    def reset(self):
        self.mutate_track(MUTATION_RANGE)
        self.calculate_center_line()
        self.car = SimpleCar(self.active_track["starting_pose_front_wing"][0],
                             self.active_track["starting_pose_front_wing"][1],
                             self.active_track["starting_pose_front_wing"][2] - pi / 2)
        self.checkpoints = deque(self.center_points)
        self.episode_step = 0
        self.last_speed_was_zero = False
        return self.get_observations()

    def get_observations(self):
        observations = list(self.checkpoints)[:INPUT_2D_SHAPE[0] - 1]
        output = [(self.car.linear_speed_value, self.car.angular_speed_value)]
        for point in observations:
            loc = self.car.location
            phi = self.car.phi + pi / 2
            x = point[0] - loc[0]
            y = point[1] - loc[1]
            x_ = x * cos(phi) - y * sin(phi)
            y_ = x * sin(phi) + y * cos(phi)
            output.append((x_, y_))

        total_points = INPUT_2D_SHAPE[0]
        while len(output) < total_points:
            output.append((0, 0))

        dt = np.dtype('float')
        return np.array(output, dtype=dt)

    def step(self, action, visualize=False):
        self.episode_step += 1
        # throttle = action[0]  # (action % 20 - 10) / 10.0
        steering = action  # [1]  # (action // 20 - 10) / 10.0
        self.car.update(self.TIME_STEP, steering)  # (throttle, steering))

        new_observation = self.get_observations()
        if self.check_track():
            reward = -self.OOB_PENALTY
            self.reset()
            self.consecutive_successes = 0
        elif self.check_checkpoints():
            reward = self.CHECKPOINT_REWARD
            # print("HIT CHECKPOINT")
        elif self.last_speed_was_zero and action[0] <= 0:
            reward = -100
            self.reset()
            self.consecutive_successes = 0
            # print("STANDING STILL")
        else:
            reward = -0.1 * (new_observation[1][0] ** 2
                             + new_observation[1][1] ** 2)  # -self.STEP_PENALTY  # self.car.linear_speed_value / 2

        if len(self.checkpoints) == 0:
            self.consecutive_successes += 1

        done = False
        if reward == -self.OOB_PENALTY \
                or self.episode_step >= EPISODE_LENGTH \
                or len(self.checkpoints) == 0 \
                or (self.last_speed_was_zero and action[0] <= 0):
            done = True

        if self.car.speed == 0:
            self.last_speed_was_zero = True

        if visualize:
            self.visualizer.render(self.active_track, self.checkpoints, self.car)
        # print("reward", reward)
        return new_observation, reward, done
