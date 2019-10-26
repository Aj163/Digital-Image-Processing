import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../images/cameraman.png', 0)
blur_img = cv.blur(img, (9, 9))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(blur_img, cmap='gray'), plt.title('Smoothened image')
plt.xticks([]), plt.yticks([])

plt.show()
