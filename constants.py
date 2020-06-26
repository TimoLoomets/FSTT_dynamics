import math

DISCOUNT = 0.99

REPLAY_MEMORY_SIZE = 500_000
MIN_REPLAY_MEMORY_SIZE = 1500
MODEL_NAME = "RELUx7"
MINIBATCH_SIZE = 1000  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 100  # Terminal states (end of episodes)
MIN_REWARD = 5

EPSILON_DECAY = 0.9925  # 0.999997  #
EPSILON_DECAYER = lambda x: -2 / (1 + math.e ** (-0.3 * x)) + 2
MIN_EPSILON = 0.05
MAX_EPSILON = 1

EPISODES = 20_000
EPISODE_LENGTH = 100
AGGREGATE_STATS_EVERY = 30  # episodes
SHOW_PREVIEW = True

MUTATION_RANGE = 0.25
SUCCESS_THRESHOLD = 7
VISUALIZE_WHILE_TRAINING = False  # True
TRACK_FILE = 'tilted_three_points.yaml'
TRACK_FILES = ['single_point.yaml',
               'tilted_single_point.yaml',
               'three_points.yaml',
               'tilted_three_points.yaml',
               'FSG.yaml']

INPUT_2D_SHAPE = (2, 2)

OUTPUT_2D_SHAPE = (5, 1)  # (10, 3)  # throttle, steering, quality
OUTPUT_1D_SHAPE = 5
ACTIONS = [(0.0, -1.0), (0.0, -0.5), (0.0, 0.0), (0.0, 0.5), (0.0, 1.0)]
'''[(-1.0, -1.0), (-1.0, -0.5), (-1.0, 0.0), (-1.0, 0.5), (-1.0, 1.0),
           (-0.5, -1.0), (-0.5, -0.5), (-0.5, 0.0), (-0.5, 0.5), (-0.5, 1.0),
           (0.0, -1.0), (0.0, -0.5), (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
           (0.5, -1.0), (0.5, -0.5), (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
           (1.0, -1.0), (1.0, -0.5), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)]'''

PIXELS_PER_METER = 8
OFFSET_X = 30 * PIXELS_PER_METER
OFFSET_Y = 80 * PIXELS_PER_METER
