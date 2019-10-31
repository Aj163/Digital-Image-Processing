import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('../images/paper.jpeg', 0)

# global thresholding with T = 180
T1, global_img = cv.threshold(img, 180, 255, cv.THRESH_BINARY)

# Otsu's thresholding
T2, otsu_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

# Local binarization with Otsu
local_otsu_img = np.copy(img)
X = img.shape[0]
Y = img.shape[1]
STEP = 50
for x in range(0, X, STEP):
    for y in range(0, Y, STEP):
        x_end = min(x + STEP, X)
        y_end = min(y + STEP, Y)
        local_img = local_otsu_img[x:x_end, y:y_end]
        T, local_img = cv.threshold(local_img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        local_otsu_img[x:x_end, y:y_end] = local_img

plt.subplot(2, 2, 1), plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.title('Global Thresholding (T=180)')
plt.imshow(global_img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.title("Otsu's Method (T=" + str(int(T2)) + ")")
plt.imshow(otsu_img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.title('Local Binarization')
plt.imshow(local_otsu_img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.show()