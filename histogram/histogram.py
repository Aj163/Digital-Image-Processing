import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../images/bridge.png', 0)

hist,bins = np.histogram(img.flatten(),256,[0,256])

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray'), plt.title('Image')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.title('Histogram')
plt.hist(img.flatten(),256,[0,256])

plt.show()