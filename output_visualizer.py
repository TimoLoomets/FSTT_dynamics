import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from interpolator import Interpolator


class OutputVisualizer:
    def __init__(self, window_name="output"):
        self.WIDTH = 250

        self.DISPLAY_STEERING_MIN = -1
        self.DISPLAY_THROTTLE_MIN = -1
        self.DISPLAY_STEERING_MAX = 1
        self.DISPLAY_THROTTLE_MAX = 1

        self.DISPLAY_STEERING_RANGE = self.DISPLAY_STEERING_MAX - self.DISPLAY_STEERING_MIN
        self.DISPLAY_THROTTLE_RANGE = self.DISPLAY_THROTTLE_MAX - self.DISPLAY_THROTTLE_MIN

        self.DISPLAY_STEP = 0.1
        self.DISPLAY_STEERING_SEGMENT_WIDTH = int(self.WIDTH * self.DISPLAY_STEP / self.DISPLAY_STEERING_RANGE)
        self.DISPLAY_THROTTLE_SEGMENT_WIDTH = int(self.WIDTH * self.DISPLAY_STEP / self.DISPLAY_THROTTLE_RANGE)

        self.HUE_RANGE = 60

        self.img = None
        self.window_name = window_name
        self._clear()
        self.interpolator = Interpolator()

    def _clear(self):
        self.img = np.zeros((self.WIDTH, self.WIDTH, 3), np.uint8)

    def _iterate(self, output):
        u = output[:, :2]
        q = output[:, 2]
        self.interpolator.set_u(u)
        self.interpolator.set_q(q)
        X = []
        Y = []
        Z = []
        for throttle in np.arange(-1, 1.1, 0.1):
            for steering in np.arange(-1, 1.1, 0.1):
                X.append(throttle)
                Y.append(steering)
                Z.append(self.interpolator.get_quality(np.array([throttle, steering])))
        return X, Y, Z

    def _coord2px(self, value):
        return int(min(max(value, 0), self.WIDTH - 1))

    def _draw_output_2(self, output):
        u = output[:, :2]
        q = output[:, 2]

        x = [action[1] for action in u]
        y = [action[0] for action in u]

        x_0 = min(x)
        y_0 = min(y)

        x_pixels_per_value = self.WIDTH / (max(x) - min(x)) if max(x) != min(x) else 0
        y_pixels_per_value = self.WIDTH / (max(y) - min(y)) if max(y) != min(y) else 0

        x_loc = [self._coord2px((x_value - x_0) * x_pixels_per_value) for x_value in x]
        y_loc = [self._coord2px(self.WIDTH - (y_value - y_0) * y_pixels_per_value) for y_value in y]

        x_values = sorted(set(x_loc))
        y_values = sorted(set(y_loc))

        x_start = dict(zip(x_values,
                           [0] + [self._coord2px((x_values[i + 1] + x_values[i]) / 2)
                                  for i in range(len(x_values) - 1)]))
        y_start = dict(zip(y_values,
                           [0] + [self._coord2px((y_values[i + 1] + y_values[i]) / 2)
                                  for i in range(len(y_values) - 1)]))

        x_stop = dict(zip(x_values,
                          [self._coord2px((x_values[i + 1] + x_values[i]) / 2)
                           for i in range(len(x_values) - 1)] + [self.WIDTH - 1]))
        y_stop = dict(zip(y_values,
                          [self._coord2px((y_values[i + 1] + y_values[i]) / 2)
                           for i in range(len(y_values) - 1)] + [self.WIDTH - 1]))

        q_0 = min(q + [0])

        hue_per_value = self.HUE_RANGE / (max(q + [0]) - min(q + [0]))

        for i in range(len(q)):
            cv2.rectangle(self.img,
                          (x_start[x_loc[i]], y_start[y_loc[i]]),
                          (x_stop[x_loc[i]], y_stop[y_loc[i]]),
                          color=tuple(map(int,
                                          cv2.cvtColor(np.uint8([[[(q[i] - q_0) * hue_per_value, 255, 255]]]),
                                                       cv2.COLOR_HSV2BGR)[0, 0])),
                          thickness=-1)

            cv2.circle(self.img,
                       (x_loc[i], y_loc[i]),
                       max(int(self.DISPLAY_THROTTLE_SEGMENT_WIDTH / 5), 1),
                       (0, 0, 0),
                       -1)

    def _draw_output(self, output):
        u = output[:, :2]
        q = output[:, 2]
        self.interpolator.set_u(u)
        self.interpolator.set_q(q)

        X = []
        Y = []
        Z = []

        for throttle in np.arange(self.DISPLAY_THROTTLE_MIN,
                                  self.DISPLAY_THROTTLE_MAX + self.DISPLAY_STEP,
                                  self.DISPLAY_STEP):
            y = (self.WIDTH - self.DISPLAY_THROTTLE_SEGMENT_WIDTH) * (0.5 - throttle / self.DISPLAY_THROTTLE_RANGE)
            for steering in np.arange(self.DISPLAY_STEERING_MIN,
                                      self.DISPLAY_STEERING_MAX + self.DISPLAY_STEP,
                                      self.DISPLAY_STEP):
                x = (self.WIDTH - self.DISPLAY_STEERING_SEGMENT_WIDTH) * (0.5 + steering / self.DISPLAY_STEERING_RANGE)

                X.append(int(x))
                Y.append(int(y))
                Z.append(self.interpolator.get_quality(np.array([throttle, steering])))

        knots = []
        for i in range(len(u)):
            throttle = u[i][0]
            steering = u[i][1]
            y = int(self.WIDTH * (0.5 - throttle / self.DISPLAY_THROTTLE_RANGE))
            x = int(self.WIDTH * (0.5 + steering / self.DISPLAY_STEERING_RANGE))
            knots.append((x, y))

        min_q = min(Z)
        range_q = max(q) - min_q

        q_multiplier = self.HUE_RANGE / range_q
        if math.isnan(q_multiplier) or math.isinf(q_multiplier):
            q_multiplier = 1

        Z = [min(max(int(q_multiplier * (z - min_q)), 0), self.HUE_RANGE) for z in Z]

        for i in range(len(Z)):
            cv2.rectangle(self.img,
                          (int(min(self.WIDTH - 1, X[i])), int(min(self.WIDTH - 1, Y[i]))),
                          (int(min(self.WIDTH - 1, X[i] + self.DISPLAY_STEERING_SEGMENT_WIDTH)),
                           int(min(self.WIDTH - 1, Y[i] + self.DISPLAY_THROTTLE_SEGMENT_WIDTH))),
                          color=tuple(map(int, cv2.cvtColor(np.uint8([[[Z[i], 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])),
                          thickness=-1)

        for knot in knots:
            cv2.circle(self.img,
                       knot,
                       max(int(self.DISPLAY_THROTTLE_SEGMENT_WIDTH / 5), 1),
                       (0, 0, 0),
                       -1)

    def render(self, output):
        self._clear()
        self._draw_output_2(output)
        cv2.imshow(self.window_name, self.img)
        cv2.waitKey(40)
