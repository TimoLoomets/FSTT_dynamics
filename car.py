from math import sin, cos
from constants import *


class Car:
    def __init__(self, x, y, phi):
        self._x = x
        self._y = y
        self._phi = phi

        self._width = 0.8
        self._height = 1.5
        # self.width = PIXELS_PER_METER * self._width
        # self.height = PIXELS_PER_METER * self._height
        self.speed = 0
        self.omega = 0
        self.speed_max = 32
        self.a_max = 7.7

    @property
    def x(self):  # m
        return self._x

    @property
    def y(self):  # m
        return self._y

    @property
    def phi(self):  # rad
        return self._phi

    @property
    def area(self):  # m^2
        return self._width * self._height

    @property
    def location(self):  # meters
        return self._x / PIXELS_PER_METER, self._y / PIXELS_PER_METER

    @property
    def angular_speed_value(self):  # rad/s
        return self.omega

    @property
    def linear_speed_value(self):  # m/s
        return self.speed

    def rotate_point(self, x, y):
        x_ = x * cos(self.phi) - y * sin(self.phi)
        y_ = x * sin(self.phi) + y * cos(self.phi)
        return x_, y_

    def get_corners(self, multiplier=1) -> list:
        points = [
            self.rotate_point(-self._width * multiplier / 2, -self._height * multiplier / 2),
            self.rotate_point(-self._width * multiplier / 2, self._height * multiplier / 2),
            self.rotate_point(self._width * multiplier / 2, self._height * multiplier / 2),
            self.rotate_point(self._width * multiplier / 2, -self._height * multiplier / 2)
        ]
        for i in range(len(points)):
            points[i] = (
                points[i][0] + self._x * multiplier, points[i][1] + self._y * multiplier)
        return points

    def get_edges(self):
        corners = self.get_corners()
        edges = []
        for corner_index in range(len(corners)):
            edges.append((corners[corner_index - 1], corners[corner_index]))
        return edges

    '''
    def get_points(self):
        points = [
            self.rotate_point(-self.width / 2, -self.height / 2),
            self.rotate_point(-self.width / 2, self.height / 2),
            self.rotate_point(self.width / 2, self.height / 2),
            self.rotate_point(self.width / 2, -self.height / 2)
        ]
        for i in range(len(points)):
            points[i] = (points[i][0] + self._x + OFFSET_X, points[i][1] + self._y + OFFSET_Y)
        return points
    '''

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
        self.move(self.speed * t)
        self.rotate(half_angle)

        self.speed *= 0.99
        self.omega *= 0.99

    def accelerate(self, throttle):
        self.speed += min(max(throttle, -1), 1) * self.a_max * (1 - self.speed / self.speed_max)
        self.speed = min(max(self.speed, 0), self.speed_max)

    def steer(self, steering):
        self.omega = min(max(steering, -1), 1) * 0.436332 * 5
