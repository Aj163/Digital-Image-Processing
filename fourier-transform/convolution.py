import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../images/cameraman.png', 0)

# Convolution
kernel = np.ones((25, 25)) / 625
conv_img = cv.filter2D(img, -1, kernel)

# Using FT
ft_filter = np.zeros(img.shape)
dim_x = img.shape[0]
dim_y = img.shape[1]
ft_filter[dim_x//2 - 12 : dim_x//2 + 13, dim_y//2 - 12 : dim_y//2 + 13] = 1/625

f_img = np.fft.fft2(img)
f_kernel = np.fft.fft2(ft_filter)
f_img = f_img * f_kernel
new_img = np.abs(np.fft.ifftshift(np.fft.ifft2(f_img)))

plt.subplot(1, 3, 1), plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.title('Convolution (Blur)')
plt.imshow(conv_img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 3), plt.title('Using Fourier Transform')
plt.imshow(new_img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.show()