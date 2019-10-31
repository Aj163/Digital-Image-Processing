import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../images/cameraman.png', 0)

f = np.fft.fft2(img)

img_filter = np.zeros(img.shape)
SIZE = 25
dim_x = img.shape[0]
dim_y = img.shape[1]
img_filter[dim_x//2 - SIZE : dim_x//2 + SIZE, dim_y//2 - SIZE : dim_y//2 + SIZE] = 1

fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

filter_fshift = fshift * img_filter
blur_f = np.fft.ifftshift(filter_fshift)
blur_img = np.fft.ifft2(blur_f)

plt.subplot(2, 2, 1), plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.title('Filter')
plt.imshow(img_filter, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 3), plt.title('Filtered Magnitude Spectrum')
plt.imshow(magnitude_spectrum * img_filter, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.title('Image on low pass filter')
plt.imshow(np.abs(blur_img), cmap='gray')
plt.xticks([]), plt.yticks([])


plt.show()