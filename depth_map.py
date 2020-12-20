import numpy as np
import cv2
from matplotlib import pyplot as plt

imgL = cv2.imread('example_left.png', 0)
imgR = cv2.imread('example_right.png', 0)

stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 15)
disparity = stereo.compute(imgL, imgR)

# f = 2
# disparity[disparity == 0] = 100000
# depth = f / disparity

# mn, mx = depth.min(), depth.max()
# depth = (depth - mn) / (mx - mn)
mn, mx = disparity.min(), disparity.max()
disparity = (disparity - mn) / (mx - mn)

plt.imshow(disparity, 'gray')
plt.show()
