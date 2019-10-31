import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/brain_MRI.jpg', 0)

# Binarize the image
ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Erode the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
erosion = cv2.erode(bin_img, kernel, iterations=1)

# Find the components
n, components = cv2.connectedComponents(erosion)
components[components != 7] = 0
mask = np.copy(components)

# Final image
brain_img = np.copy(img)
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        if not mask[x][y]:
            brain_img[x][y] = 0

plt.subplot(2, 2, 1), plt.title('Original image')
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.title('Eroded image')
plt.imshow(erosion, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.title('Mask')
plt.imshow(mask, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.title('Segmented Brain')
plt.imshow(brain_img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.show()