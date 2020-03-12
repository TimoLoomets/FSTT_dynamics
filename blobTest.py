import numpy as np
import cv2
from network import Blob
from network import BlobEnv
from PIL import Image

env = BlobEnv()
env.reset()
env.render()
open_cv_image = np.array(env.get_image())
open_cv_image = open_cv_image[:, :, ::-1].copy()
cv2.imshow("img", open_cv_image)
cv2.waitKey(0)
