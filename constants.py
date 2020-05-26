DISCOUNT = 0.5

REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 256
MODEL_NAME = "RELUx3"
MINIBATCH_SIZE = 128  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 10  # Terminal states (end of episodes)
MIN_REWARD = 5

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.1

EPISODES = 10_000
EPISODE_LENGTH = 200
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True

INPUT_2D_SHAPE = (6, 2)

OUTPUT_2D_SHAPE = (25, 1)  # (10, 3)  # throttle, steering, quality
OUTPUT_1D_SHAPE = 25
ACTIONS = [(-1.0, -1.0), (-1.0, -0.5), (-1.0, 0.0), (-1.0, 0.5), (-1.0, 1.0),
           (-0.5, -1.0), (-0.5, -0.5), (-0.5, 0.0), (-0.5, 0.5), (-0.5, 1.0),
           (0.0,  -1.0), (0.0,  -0.5), (0.0,  0.0), (0.0,  0.5), (0.0,  1.0),
           (0.5,  -1.0), (0.5,  -0.5), (0.5,  0.0), (0.5,  0.5), (0.5,  1.0),
           (1.0,  -1.0), (1.0,  -0.5), (1.0,  0.0), (1.0,  0.5), (1.0,  1.0)]

PIXELS_PER_METER = 8
OFFSET_X = 30 * PIXELS_PER_METER
OFFSET_Y = 80 * PIXELS_PER_METER
