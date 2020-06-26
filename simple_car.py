from car import Car


class SimpleCar(Car):
    def __init__(self, x, y, phi):
        super().__init__(x, y, phi)

        self.CONSTANT_CAR_SPEED = 1

        self.speed = self.CONSTANT_CAR_SPEED

    def update(self, t, control):
        self.steer(control[1])

        half_angle = self.omega / 2 * t * (self.speed / self.speed_max)
        self.rotate(half_angle)
        self.move(self.speed * t)
        self.rotate(half_angle)

        self.omega *= 0.99
