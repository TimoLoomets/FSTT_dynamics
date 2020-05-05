import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from interpolator import Interpolator


class OutputVisualizer:
    def __init__(self):
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
        self._draw_output(output)
        cv2.imshow('output', self.img)
        cv2.waitKey(40)
