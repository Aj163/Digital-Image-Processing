import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('../images/person.jpeg')

plt.subplot(1, 2, 1), plt.title('Original Image')
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])

new_img = img
for x in range(new_img.shape[0]):
    for y in range(new_img.shape[1]):
        b = int(new_img[x][y][0])
        g = int(new_img[x][y][1])
        r = int(new_img[x][y][2])
        
        if g > 45:
            avg = (r + g + b) // 3
            new_img[x][y][0] = avg
            new_img[x][y][1] = avg
            new_img[x][y][2] = avg

plt.subplot(1, 2, 2), plt.title('Transformed Image')
plt.imshow(cv.cvtColor(new_img, cv.COLOR_BGR2RGB))
plt.xticks([]), plt.yticks([])

plt.show()