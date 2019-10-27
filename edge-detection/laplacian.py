import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../images/cameraman.png', 0)

orig_img = np.array(img)
img = cv.GaussianBlur(img, (7, 7), 0)
edge_img = np.uint8(cv.Laplacian(img, cv.CV_16S))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray'), plt.title('Original')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(edge_img, cmap='gray'), plt.title('Laplacian')
plt.xticks([]), plt.yticks([])

plt.show()
