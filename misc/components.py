import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/circles.jpeg', 0)

# Binarize the image
ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Erode the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
erosion = cv2.erode(img, kernel, iterations=5)

# Find the components
n, components = cv2.connectedComponents(erosion)

plt.subplot(1, 3, 1), plt.title('Original image')
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.title('Eroded image')
plt.imshow(erosion, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.title('Components')
plt.imshow(components, cmap='nipy_spectral')
plt.xticks([]), plt.yticks([])

print('Number of circles:', n-1)
plt.show()