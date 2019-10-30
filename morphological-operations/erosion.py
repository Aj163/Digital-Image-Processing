import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/j.png', 0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
erosion = cv2.erode(img, kernel)

plt.subplot(1, 2, 1), plt.title('Original image')
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.title('Eroded image')
plt.imshow(erosion, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.show()