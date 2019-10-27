import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../images/cameraman.png', 0)
kernel = np.ones((9, 9), np.float32) / 81

orig_img = np.array(img)
img = cv.GaussianBlur(img, (7, 7), 0)
sobel_x = (np.abs(cv.Sobel(img, cv.CV_16S, 1, 0, ksize=5)))
sobel_y = (np.abs(cv.Sobel(img, cv.CV_16S, 0, 1, ksize=5)))
sobel = ((np.uint16(sobel_x)**2 + (np.uint16(sobel_y)**2)**0.5))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(sobel, cmap='gray'), plt.title('Sobel Y')
plt.xticks([]), plt.yticks([])

plt.show()
