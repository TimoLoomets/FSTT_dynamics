import numpy as np
import cv2


class InputVisualizer:
    def __init__(self):
        self.MIN_WIDTH = 600
        self.PIXELS_PER_METER = 150

        self.img = None
        self._clear()

    def _clear(self):
        self.img = np.zeros((self.MIN_WIDTH, self.MIN_WIDTH, 3), np.uint8)

    def _draw_input(self, input):
        odom = input[0]
        cones = list(filter(lambda x: x[0] != 0 or x[1] != 0, input[1:]))

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
        negative_x = int(max(min_x * self.PIXELS_PER_METER, self.MIN_WIDTH / 2))
        positive_x = int(max(max_x * self.PIXELS_PER_METER, self.MIN_WIDTH / 2))

        for cone in cones:
            loc = (int(cone[0] / max_x * self.MIN_WIDTH / 2), cone[1])
            cv2.circle(self.img,
                       cone,
                       max(int(self.DISPLAY_THROTTLE_SEGMENT_WIDTH / 5), 1),
                       (0, 0, 0),
                       -1)
