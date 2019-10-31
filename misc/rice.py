import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/rice.jpeg', 0)
orig_img = np.copy(img)

# Binarize the image
X = img.shape[0]
Y = img.shape[1]
STEP = 70
for x in range(0, X, STEP):
    for y in range(0, Y, STEP):
        x_end = min(x + STEP, X)
        y_end = min(y + STEP, Y)
        local_img = img[x:x_end, y:y_end]
        T, local_img = cv2.threshold(local_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img[x:x_end, y:y_end] = local_img

# Erode the image
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
erosion = cv2.erode(img, kernel, iterations=1)

# Find the components
n, components = cv2.connectedComponents(erosion)

plt.subplot(1, 3, 1), plt.title('Original image')
plt.imshow(orig_img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.title('Eroded image')
plt.imshow(erosion, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.title('Components')
plt.imshow(components, cmap='nipy_spectral')
plt.xticks([]), plt.yticks([])

print('Number of rice grains:', n-1)
plt.show()