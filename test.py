import math
import pyglet
from pyglet.window import key
import numpy as np
from pyglet.gl import *

win = pyglet.window.Window()
batch = pyglet.graphics.Batch()
white = [255]*4

def draw_polygon():
    global batch
    batch.add(4, pyglet.gl.GL_POLYGON, None, ('v2i', [10, 60, 10, 110, 390, 60, 390, 110]), ('c4B', white * 4))

class Car:
    def __init__(self, x, y, phi):
        self._x = x
        self._y = y
        self._phi = phi

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def phi(self):
        return self._phi


x1 = 100
x2 = 25
y1 = 0
y2 = 25

# function that increments to the next
# point along a circle
frame = 0


def rect(x, y, width, height):
    width = int(round(width))
    height = int(round(height))
    image_pattern = pyglet.image.SolidColorImagePattern(color=(255, 255, 255, 255))
    image = image_pattern.create_image(width, height)
    image.blit(x, y)


@win.event
def on_draw():
    # clear the screen
    glClear(GL_COLOR_BUFFER_BIT)
    # draw the next line
    # in the circle animation
    # circle centered at 100,100,0 = x,y,z
    global x1, y1, x2, y2, batch

    rect(x1, y1, x2, y2)
    draw_polygon()
    batch.draw()


@win.event
def on_key_press(symbol, modifiers):
    global x1, y1, x2, y2
    rect(x1, y1, x2, y2)

    if symbol == key.LEFT:
        x1 = x1 - 10
        rect(x1, y1, x2, y2)
    elif symbol == key.RIGHT:
        x1 = x1 + 10
        rect(x1, y1, x2, y2)
    elif symbol == key.UP:
        y1 = y1 - 10
        rect(x1, y1, x2, y2)
    elif symbol == key.DOWN:
        y1 = y1 + 10
        rect(x1, y1, x2, y2)


pyglet.app.run()
