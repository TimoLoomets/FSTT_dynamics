import numpy as np
import cv2

from constants import *


class Visualizer:
    def __init__(self):
        self.img = np.zeros((700, 700, 3), np.uint8)

    @staticmethod
    def point_to_coord(point):
        return round(PIXELS_PER_METER * point[0]) + OFFSET_X, round(
            PIXELS_PER_METER * point[1]) + OFFSET_Y

    def clear(self):
        self.img = np.zeros((700, 700, 3), np.uint8)

    def draw_cone(self, cone, color):
        center = cone
        cv2.circle(self.img, center, 2, color, -1)

    def draw_car(self, car):
        pts = np.array([Visualizer.point_to_coord(corner) for corner in car.get_corners()], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(self.img, [pts], True, (0, 0, 255))

    def draw_track(self, track, checkpoints):
        last_cone = None
        for cone in track["cones_left"]:
            cone = Visualizer.point_to_coord(cone)
            self.draw_cone(cone, (255, 0, 0))
            if last_cone is not None:
                cv2.line(self.img, last_cone, cone, (255, 0, 0), 1)
            last_cone = cone
        last_cone = None
        for cone in track["cones_right"]:
            cone = Visualizer.point_to_coord(cone)
            self.draw_cone(cone, (0, 255, 255))
            if last_cone is not None:
                cv2.line(self.img, last_cone, cone, (0, 255, 255), 1)
            last_cone = cone

        last_cone = None
        for cone in checkpoints:
            cone = Visualizer.point_to_coord(cone)
            self.draw_cone(cone, (0, 255, 0))
            if last_cone is not None:
                cv2.line(self.img, last_cone, cone, (0, 255, 0), 1)
            last_cone = cone

    def render(self, track, checkpoints, car):
        self.clear()
        self.draw_track(track, checkpoints)
        self.draw_car(car)
        cv2.imshow('image', self.img)
        cv2.waitKey(40)
