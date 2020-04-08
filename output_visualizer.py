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
        self.clear()
        self.interpolator = Interpolator()

    def clear(self):
        self.img = np.zeros((self.WIDTH, self.WIDTH, 3), np.uint8)

    def iterate(self, output):
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

    def draw_output(self, output):
        throttles, steering, qualities = self.iterate(output)
        throttles = np.reshape(throttles, (-1, 7))
        steerings = np.reshape(steering, (-1, 7))
        qualities = np.reshape(qualities, (-1, 7))

        ax = plt.axes()
        ax.contourf(np.array(throttles), np.array(steerings), np.array(qualities), cmap=cm.bwr)
        # ax.set_ylim([-1, 1])
        # ax.set_xlim([-1, 1])

        canvas = plt.gcf().canvas
        canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (h, w, 4)
        self.img = cv2.cvtColor(np.roll(buf, 3, axis=2)[:, :, :3], cv2.COLOR_RGB2BGR)
        # plt.show()
        plt.clf()

    def draw_output_2(self, output):
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
            y = self.WIDTH * (0.5 - throttle / self.DISPLAY_THROTTLE_RANGE)
            for steering in np.arange(self.DISPLAY_STEERING_MIN,
                                      self.DISPLAY_STEERING_MAX + self.DISPLAY_STEP,
                                      self.DISPLAY_STEP):
                x = self.WIDTH * (0.5 + steering / self.DISPLAY_STEERING_RANGE)

                X.append(int(x))
                Y.append(int(y))
                Z.append(self.interpolator.get_quality(np.array([throttle, steering])))

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

    def render(self, output):
        self.clear()
        self.draw_output_2(output)
        cv2.imshow('output', self.img)
        cv2.waitKey(40)
