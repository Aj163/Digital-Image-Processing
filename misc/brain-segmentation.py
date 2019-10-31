import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/brain_MRI.jpg', 0)

# Binarize the image
ret, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Erode the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
erosion = cv2.erode(bin_img, kernel, iterations=1)

# Find the components
n, mask = cv2.connectedComponents(erosion)
mask[mask != 4] = 0
mask[mask != 0] = 1

mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

# Final image
brain_img = img * mask

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