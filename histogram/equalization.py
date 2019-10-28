import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('../images/bridge.png', 0)
equ = cv2.equalizeHist(img)
res = np.hstack((img,equ)) #stacking images side-by-side

plt.title('Original Image vs Enhanced image')
plt.imshow(res, cmap='gray')
plt.xticks([]), plt.yticks([])

plt.show()
