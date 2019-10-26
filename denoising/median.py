import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../images/salt-and-pepper.png', 0)
blur_img = cv.medianBlur(img, 3)

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(blur_img, cmap='gray'), plt.title('Denoised image')
plt.xticks([]), plt.yticks([])

plt.show()
