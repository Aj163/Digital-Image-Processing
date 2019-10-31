import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('../images/cameraman.png', 0)

f = np.fft.fft2(img)
o_img = np.fft.ifft2(f)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

plt.subplot(1, 2, 1), plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2), plt.title('Magnitude Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.show()