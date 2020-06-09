DISCOUNT = 0.95

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 256
MODEL_NAME = "RELUx3"
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)
MIN_REWARD = 5

EPSILON_DECAY = 0.97
MIN_EPSILON = 0.05

EPISODES = 1000
EPISODE_LENGTH = 20
AGGREGATE_STATS_EVERY = 20  # episodes
SHOW_PREVIEW = True

MUTATION_RANGE = 0.25
VISUALIZE_WHILE_TRAINING = False  # True
TRACK_FILE = 'single_point.yaml'

INPUT_2D_SHAPE = (3, 2)

OUTPUT_2D_SHAPE = (25, 1)  # (10, 3)  # throttle, steering, quality
OUTPUT_1D_SHAPE = 25
ACTIONS = [(-1.0, -1.0), (-1.0, -0.5), (-1.0, 0.0), (-1.0, 0.5), (-1.0, 1.0),
           (-0.5, -1.0), (-0.5, -0.5), (-0.5, 0.0), (-0.5, 0.5), (-0.5, 1.0),
           (0.0, -1.0), (0.0, -0.5), (0.0, 0.0), (0.0, 0.5), (0.0, 1.0),
           (0.5, -1.0), (0.5, -0.5), (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
           (1.0, -1.0), (1.0, -0.5), (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)]

PIXELS_PER_METER = 8
OFFSET_X = 30 * PIXELS_PER_METER
OFFSET_Y = 80 * PIXELS_PER_METER
