import numpy as np
import cv2


class InputVisualizer:
    def __init__(self):
        self.MIN_WIDTH = 600
        self.PIXELS_PER_METER = 15

        self.img = None
        self._clear()

    def _clear(self, width=None, height=None):
        if width is None or height is None:
            self.img = np.zeros((self.MIN_WIDTH, self.MIN_WIDTH, 3), np.uint8)
        else:
            self.img = np.zeros((width, height, 3), np.uint8)

    def _draw_input(self, input):
        odom = input[0]
        cones = list(filter(lambda x: x[0] != 0 or x[1] != 0, input[1:]))
        print(cones)

        last_loc = (int(self.MIN_WIDTH / 2), int(self.MIN_WIDTH / 2))
        cv2.circle(self.img,
                   last_loc,
                   max(int(self.MIN_WIDTH / 100), 1),
                   (255, 0, 0),
                   -1)

        max_x = max(cone[0] for cone in cones)
        min_x = min(cone[0] for cone in cones)
        max_y = max(cone[1] for cone in cones)
        min_y = min(cone[1] for cone in cones)

        positive_x = int(max(max_x * self.PIXELS_PER_METER, self.MIN_WIDTH / 2))
        negative_x = int(max(-min_x * self.PIXELS_PER_METER, self.MIN_WIDTH / 2))
        positive_y = int(max(max_y * self.PIXELS_PER_METER, self.MIN_WIDTH / 2))
        negative_y = int(max(-min_y * self.PIXELS_PER_METER, self.MIN_WIDTH / 2))

        self._clear(int(positive_x + negative_x), int(positive_y + negative_y))

        for cone in cones:
            loc = (int(cone[0] * self.PIXELS_PER_METER), int(cone[1] * self.PIXELS_PER_METER))
            cv2.circle(self.img,
                       loc,
                       max(int(self.MIN_WIDTH / 100), 1),
                       (255, 0, 0),
                       -1)
            cv2.line(self.img,
                     last_loc,
                     loc,
                     (255, 0, 0),
                     2)
            last_loc = loc

    def render(self, input):
        self._draw_input(input)
        cv2.imshow('input', self.img)
        cv2.waitKey(40)
