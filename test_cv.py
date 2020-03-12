import cv2
import numpy as np
from math import cos
from math import sin
from math import pi
import yaml
from os import path
import time
from collections import deque

from cone_filter_sorter_node import ConeFilterNode

size = 100
img = np.zeros((700, 700, 3), np.uint8)
pixels_per_meter = 8
offset_x = 30 * pixels_per_meter
offset_y = 80 * pixels_per_meter
track = {}
time_factor = 1
sorter = ConeFilterNode()
center_points = []
checkpoints = deque()


class Car:
    def __init__(self, x, y, phi):
        self._x = x
        self._y = y
        self._phi = phi

        self._width = 0.8
        self._height = 1.5
        self.width = pixels_per_meter * self._width
        self.height = pixels_per_meter * self._height
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
        return self._x / pixels_per_meter, self._y / pixels_per_meter

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
            points[i] = (points[i][0] + self._x / pixels_per_meter, points[i][1] + self._y / pixels_per_meter)
        return points

    def get_points(self):
        points = [
            self.rotate_point(-self.width / 2, -self.height / 2),
            self.rotate_point(-self.width / 2, self.height / 2),
            self.rotate_point(self.width / 2, self.height / 2),
            self.rotate_point(self.width / 2, -self.height / 2)
        ]
        for i in range(len(points)):
            points[i] = (points[i][0] + self._x + offset_x, points[i][1] + self._y + offset_y)
        return points

    def draw(self):
        pts = np.array(self.get_points(), np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], True, (0, 0, 255))

    def move(self, distance):
        movement = self.rotate_point(0, distance)
        self._x += movement[0]
        self._y += movement[1]

    def rotate(self, angle):
        self._phi += angle

    def update(self, t):
        half_angle = self.omega / 2 * t
        self.rotate(half_angle)
        self.move(self.speed * t * pixels_per_meter)
        self.rotate(half_angle)

        self.speed *= 0.99
        self.omega *= 0.99

    def accelerate(self, throttle):
        self.speed += min(max(throttle, -1), 1) * self.a_max * (1 - self.speed / self.speed_max)
        self.speed = min(max(self.speed, 0), self.speed_max)

    def steer(self, steering):
        self.omega = min(max(steering, -1), 1) * 0.436332 * 5


def clear():
    global img
    img = np.zeros((700, 700, 3), np.uint8)


def point_to_coord(point):
    return round(pixels_per_meter * point[0]) + offset_x, round(pixels_per_meter * point[1]) + offset_y


def draw_cone(cone, color):
    center = cone
    cv2.circle(img, center, 2, color, -1)


def check_track():
    corners = car.get_corners()
    area = car.area
    for cone in track["cones_left"]:
        if point_in_rectangle(cone, corners, area):
            reset()
    for cone in track["cones_right"]:
        if point_in_rectangle(cone, corners, area):
            reset()


def check_checkpoints():
    global checkpoints
    corners = car.get_corners()
    area = car.area
    for i in range(2):
        if point_in_rectangle(checkpoints[i], corners, area):
            for j in range(i + 1):
                checkpoints.popleft()
            break


def triangle_area(A, B, C):
    return abs((B[0] * A[1] - A[0] * B[1]) + (C[0] * B[1] - B[0] * C[1]) + (A[0] * C[1] - C[0] * A[1])) / 2


def point_in_rectangle(point, rectangle, rectangle_area):
    total_area = 0
    for i in range(len(rectangle)):
        total_area += triangle_area(point, rectangle[i], rectangle[i - 1])
    return abs(total_area - rectangle_area) < 0.001


def load_track():
    global track
    track = yaml.load(open(path.join('FSG.yaml'), 'r'), Loader=yaml.FullLoader)
    print(track)


def calculate_center_line():
    sorter.pose_update(car)
    sorted_pairs = sorter.map_update((track["cones_right"], track["cones_left"]))
    print("yellows:", sorted_pairs.yellowCones)
    print("blues:", sorted_pairs.blueCones)
    # print(len(sorted_pairs.yellowCones), len(sorted_pairs.blueCones))
    for i in range(len(sorted_pairs.yellowCones)):
        left_cone = sorted_pairs.blueCones[i]
        right_cone = sorted_pairs.yellowCones[i]
        center_points.append(((left_cone.x + right_cone.x) / 2, (left_cone.y + right_cone.y) / 2))


def draw_track():
    last_cone = None
    for cone in track["cones_left"]:
        cone = point_to_coord(cone)
        draw_cone(cone, (255, 0, 0))
        if last_cone is not None:
            cv2.line(img, last_cone, cone, (255, 0, 0), 1)
        last_cone = cone
    last_cone = None
    for cone in track["cones_right"]:
        cone = point_to_coord(cone)
        draw_cone(cone, (0, 255, 255))
        if last_cone is not None:
            cv2.line(img, last_cone, cone, (0, 255, 255), 1)
        last_cone = cone

    last_cone = None
    # col = 255
    for cone in checkpoints:
        cone = point_to_coord(cone)
        draw_cone(cone, (0, 255, 0))
        # col = round(0.98 * col)
        if last_cone is not None:
            cv2.line(img, last_cone, cone, (0, 255, 0), 1)
        last_cone = cone


def reset():
    global car, checkpoints
    car = Car(track["starting_pose_front_wing"][0], track["starting_pose_front_wing"][1],
              track["starting_pose_front_wing"][2] - pi / 2)
    checkpoints = deque(center_points)


load_track()
car = Car(track["starting_pose_front_wing"][0], track["starting_pose_front_wing"][1],
          track["starting_pose_front_wing"][2] + pi / 2)
calculate_center_line()

reset()
last_time = time.time()
while 1:
    time_now = time.time()
    clear()
    draw_track()

    delta_time = time_now - last_time
    last_time = time_now
    car.update(delta_time * time_factor)
    car.draw()

    cv2.imshow('image', img)

    check_track()
    check_checkpoints()

    k = cv2.waitKey(10)
    if k == 27:  # Esc key to stop
        break
    elif k == 82:
        car.accelerate(0.1)
    elif k == 84:
        car.accelerate(-0.1)

    if k == 81:
        car.steer(-1)
    elif k == 83:
        car.steer(1)
    else:
        car.steer(0)
# cv2.destroyAllWindows()
