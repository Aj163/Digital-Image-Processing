import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../images/sudoku.jpg', 0)
kernel = np.ones((9, 9), np.float32) / 81

orig_img = np.array(img)
img = cv.GaussianBlur(img, (9, 9), 0)
blur_img = cv.blur(img, (3, 3))
edge_img = img - blur_img

plt.subplot(1, 2, 1)
plt.imshow(orig_img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(edge_img, cmap='gray'), plt.title('Subtracting smoothened image')
plt.xticks([]), plt.yticks([])

plt.show()
