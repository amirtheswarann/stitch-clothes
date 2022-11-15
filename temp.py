from contour import *
from imageProcessing import *
from matplotlib import pyplot as plt

import cv2

img = cv2.imread('input/1664957300391_Snap.jpg')
# img = img[168:2500,844:5150]
img = green_screen(img, get_mask(img),  green2alpha=False)
plt.imshow(img)
plt.show()
img =  img[476:3090,1245:4701]
# keep_only_major(img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
keep_only_major_histogram(img)

git config --global user.email "www.amirtheswaran4122@gmail.com"
git config --global user.name "Amirtheswaran"