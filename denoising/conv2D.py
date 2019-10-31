import cv2
import numpy as np 
from matplotlib import pyplot as plt 

img = cv2.imread('../images/cameraman.png', 0)

kernel = np.zeros(img.shape)
r, c = kernel.shape

x = cv2.getGaussianKernel(15, 10)
gaussian = x * x.T

kernel[r//2 - 7 : r//2 + 8, c//2 - 7 : c//2 + 8] = gaussian
blur_img = cv2.filter2D(img, -1, gaussian)

plt.subplot(121), plt.title('Image')
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(122), plt.title('Blurred Image')
plt.imshow(blur_img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.show()