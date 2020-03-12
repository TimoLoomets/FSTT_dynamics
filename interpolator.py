import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d


class Interpolator:
    def __init__(self):
        DEFAULT_STEP = 1  # 10 ** 5

        self.c = 0.9  # Smoothing factor
        self.e = 0.01  # sys.float_info.epsilon  # Really small number

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
        return self.wsum(action) / self.norm(action)

    def update_function(self, action, quality):
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

        self.actions = action
        self.qualities = q

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
    u = []
    interpolator = Interpolator()
    for i in np.arange(1, -0.1, -0.5):
        for j in np.arange(-1, 1.1, 0.5):
            u.append(np.array([i, j]))
    q = [0.2, 0.65, 1, 0.65, 0.2,
         0.15, 0.25, 0.5, 0.25, 0.15,
         0, 0, 0, 0, 0]

    interpolator.set_q(q)
    interpolator.set_u(u)

    # print(interpolator.get_quality(np.array([0.75, 0])))

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    X = []
    Y = []
    Z = []
    for throttle in np.arange(-1, 1.1, 0.1):
        for steering in np.arange(-1, 1.1, 0.1):
            X.append(throttle)
            Y.append(steering)
            Z.append(interpolator.get_quality(np.array([throttle, steering])))
    # ax.plot_trisurf(np.array(X), np.array(Y), np.array(Z), cmap=cm.bwr)

    throttles = [a[0] for a in u]
    steerings = [a[1] for a in u]
    # ax.plot_trisurf(np.array(throttles), np.array(steerings), np.array(q))

    interpolator.update_function(np.array([1, 0]), 2)
    interpolator.update_function(np.array([1, 0]), 2)
    interpolator.update_function(np.array([1, 0]), 2)
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

    u = interpolator.get_u()
    q = interpolator.get_q()
    throttles = [a[0] for a in u]
    steerings = [a[1] for a in u]
    ax.plot_trisurf(np.array(throttles), np.array(steerings), np.array(q), cmap=cm.bwr)

    plt.show()
