import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import math


class Interpolator:
    def __init__(self):
        DEFAULT_STEP = 1  # 10 ** 5

        self.c = 0.9  # Smoothing factor
        self.e = 0.1  # sys.float_info.epsilon  # Really small number

        self.actions = []
        self.qualities = []
        self.knots_count = 0
        self.step = DEFAULT_STEP

    def distance(self, chosen_action, i, q_max):
        return np.linalg.norm(np.subtract(chosen_action, self.actions[i])) ** 2 + self.c * (
                q_max - self.qualities[i]) + self.e

    def wsum(self, chosen_action):
        output = 0
        q_max = max(self.qualities)
        for i in range(self.knots_count):
            output += self.qualities[i] / self.distance(chosen_action, i, q_max)
        return output

    def norm(self, chosen_action):
        output = 0
        q_max = max(self.qualities)
        for i in range(self.knots_count):
            output += 1 / self.distance(chosen_action, i, q_max)
        return output

    def get_quality(self, action):
        value = self.wsum(action) / self.norm(action)
        if math.isnan(value):
            return 0
        else:
            return value

    def update_function_2(self, action, quality, update_action=True):
        q = np.array(self.qualities)

        knot_count = len(q)

        optimal_action = action
        action = np.array(self.actions)
        Q_new = quality

        num = 0
        den = 0
        deriv_q = []
        deriv_u0 = []
        deriv_u1 = []

        for it in range(0, knot_count):
            weight = np.linalg.norm(optimal_action - action[it]) + self.c * (q.max() - q[it] + self.e)
            den = den + (1.0 / weight)
            num = num + (q[it] / weight)
            deriv_q.append((den * (weight + q[it] * self.c) - num * self.c) / pow((weight * den), 2))
            deriv_u0.append(((num - den * q[it]) * 2 * (action[it][0] - optimal_action[0])) / (pow(weight * den, 2)))
            deriv_u1.append(((num - den * q[it]) * 2 * (action[it][1] - optimal_action[1])) / (pow(weight * den, 2)))

        Q_dash = num / den
        error = Q_new - Q_dash

        for it in range(0, knot_count):
            q[it] = q[it] + error * deriv_q[it]
            action[it][0] = action[it][0] + error * deriv_u0[it]
            action[it][1] = action[it][1] + error * deriv_u1[it]

        if update_action:
            self.actions = action
        self.qualities = q

    def update_function(self, action, quality, update_action=False):
        knot_count = len(self.qualities)
        # print("qualities:", self.qualities)
        if type(self.qualities) == np.ndarray:
            self.qualities = self.qualities.tolist()
        if type(self.qualities[0]) == list:
            self.qualities = [e[0] for e in self.qualities]
        max_list = self.qualities + [float(quality)]

        q_max = max(max_list)
        for it in range(0, knot_count):
            self.qualities[it] += self.e * \
                                  (quality - self.qualities[it]) \
                                  / self.distance(action, it, q_max) ** 2

    def set_u(self, actions):
        self.actions = actions
        self.knots_count = len(self.actions)

    def set_q(self, qualities):
        self.qualities = qualities

    def set_step(self, step):
        self.step = step

    def get_u(self):
        return self.actions

    def get_q(self):
        return self.qualities


if __name__ == "__main__":
    from output_visualizer import OutputVisualizer
    import cv2

    u = []
    interpolator = Interpolator()
    for i in np.arange(-1, 1.1, 0.5):
        for j in np.arange(-1, 1.1, 0.5):
            u.append(np.array([i, j]))
    q = [0.04448929, 0.5086165, 0.76275706, -0.2851543, 0.39455223,
         -0.19585085, -0.52812827, 0.25080782, 0.4987614, 0.26595366,
         -0.3598364, 0.41622806, 0.10484912, -0.11532316, -0.11455766,
         -0.14297369, -0.04747943, 0.19820265, 0.5723205, 0.13500524,
         -0.24156858, 0.15854892, 0.22840545, 0.35542938, -0.5061423]

    visualizer = OutputVisualizer()
    visualizer.render(np.append(u, [[e] for e in q], axis=1))
    cv2.waitKey(3000)

    interpolator.set_q(q)
    interpolator.set_u(u)

    # for _ in range(5):
    #    interpolator.update_function_2(np.array([0, 0]), 2)  # , update_action=False)
    # interpolator.update_function(np.array([-1, 0]), 2)#, update_action=False)
    interpolator.update_function(np.array([-0.5, 1.0]), -0.6402964293956757)  # , update_action=False)

    q = interpolator.get_q()
    u = interpolator.get_u()
    visualizer.render(np.append(u, [[e] for e in q], axis=1))
    cv2.waitKey(3000)

    # print(interpolator.get_quality(np.array([0.75, 0])))
    '''
    fig = plt.figure()
    ax = plt.axes()  # projection="3d")

    X = []
    Y = []
    Z = []
    for throttle in np.arange(-1, 1.1, 0.1):
        for steering in np.arange(-1, 1.1, 0.1):
            X.append(throttle)
            Y.append(steering)
            Z.append(interpolator.get_quality(np.array([throttle, steering])))
    '''
    # ax.plot_trisurf(np.array(X), np.array(Y), np.array(Z), cmap=cm.bwr)

    # throttles = [a[0] for a in u]
    # steerings = [a[1] for a in u]
    # ax.plot_trisurf(np.array(throttles), np.array(steerings), np.array(q))

    # interpolator.update_function(np.array([1, 0]), 20)
    # interpolator.update_function(np.array([1, 0]), 20)

    '''
    X = []
    Y = []
    Z = []
    for throttle in np.arange(-1, 1.1, 0.1):
        for steering in np.arange(-1, 1.1, 0.1):
            X.append(throttle)
            Y.append(steering)
            Z.append(interpolator.get_quality(np.array([throttle, steering])))
    ax.plot_trisurf(np.array(X), np.array(Y), np.array(Z), cmap=cm.bwr)
    '''

    '''
    u = interpolator.get_u()
    q = np.reshape(interpolator.get_q(), (-1, 5))
    throttles = np.reshape([a[0] for a in u], (-1, 5))
    steerings = np.reshape([a[1] for a in u], (-1, 5))
    ax.contourf(np.array(throttles), np.array(steerings), np.array(q), cmap=cm.bwr)

    plt.show()
    '''
