import time
start = time.time()
from functions import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio

img1 = cv2.imread('input/snap1_Mark.jpg')
img2 = cv2.imread('input/snap2_Mark.jpg')

img1 = img1[168:2500,844:5150]
img2 = img2[168:2500,844:5150]
m1 = greenScreen(img1)
m2 = greenScreen(img2)
# convert to m1 and m2 to gray
# m1 = cv2.cvtColor(m1, cv2.COLOR_BGR2GRAY)
# m2 = cv2.cvtColor(m2, cv2.COLOR_BGR2GRAY)

show2img(m1, m2)

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
th1, threshed1 = cv2.threshold(gray1, 255, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
th2, threshed2 = cv2.threshold(gray2, 255, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
threshed1 = removeWits(threshed1)
threshed2 = removeWits(threshed2)
cnts1 = cv2.findContours(threshed1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts2 = cv2.findContours(threshed2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

cnts1 = cnts1[0:2]

# show2img(img1, img2)
cPoints1 = []
cPoints2 = []
for c in cnts1:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cPoints1.append([cX, cY])
for c in cnts2:
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cPoints2.append([cX, cY])

for i in range(len(cPoints1)):
    cv2.circle(img1, (cPoints1[i][0], cPoints1[i][1]), 3, (0, 0, 255), -1)
    cv2.circle(img2, (cPoints2[i][0], cPoints2[i][1]), 3, (0, 0, 255), -1)

l1, r1 = left_right(cPoints1[0], cPoints1[1])
l2, r2 = left_right(cPoints2[0], cPoints2[1])

def getAngle(p1, p2):
    yDiff = p2[0] - p1[0]
    xDiff = p2[1] - p1[1]
    return np.degrees(np.arctan2(yDiff, xDiff))

def rotate_bound(image, angle, center):
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

angle1 = 90 - getAngle(l1, r1)
rimg1 = rotate_bound(img1, angle1, l1)
rimg1 = rimg1[0:l1[1], 0:rimg1.shape[1]]

m1 = rotate_bound(m1, angle1, l1)
m1 = m1[0:l1[1], 0:m1.shape[1]]
angle2 = 90 - getAngle(l2, r2)
rimg2 = rotate_bound(img2, angle2, l2)
rimg2 = rimg2[l2[1]:rimg2.shape[0], 0:rimg2.shape[1]]

m2 = rotate_bound(m2, angle2, l2)
m2 = m2[l2[1]:m2.shape[0], 0:m2.shape[1]]
cv2.imwrite('output/m21.jpg', m2)
diff = l1[0] - l2[0]
if diff ==0:
    pass
elif diff > 0:
    # create a green image with the same size as the image 
    # to be pasted  
    black2 = np.zeros((rimg2.shape[0], diff, 3), dtype=np.uint8)
    mm2 = black2[:,:,0]
    black1 = np.zeros((rimg1.shape[0], diff, 3), dtype=np.uint8)
    mm1 = black1[:,:,0]
    rimg2 = np.concatenate((black2, rimg2), axis=1)
    m2 = np.concatenate((black2, m2), axis=1)
    rimg1 = np.concatenate((rimg1, black1), axis=1)
    # m1 = np.concatenate((m1, black1), axis=1)
else:
    black1 = np.zeros((rimg1.shape[0], abs(diff), 3), dtype=np.uint8)
    mm1 = black1[:,:,0]
    black2 = np.zeros((rimg2.shape[0], abs(diff), 3), dtype=np.uint8)
    mm2 = black2[:,:,0]
    rimg1 = np.concatenate((black1, rimg1), axis=1)
    # conver mask to black and white image
    
    m1 = np.concatenate((mm1, m1), axis=1)
    rimg2 = np.concatenate((rimg2, black2), axis=1)
    m2 = np.concatenate((m2, mm2), axis=1)

newimg = np.concatenate((rimg1, rimg2), axis=0)
newm = np.concatenate((m1, m2), axis=0)
thresh = newm

sz=thresh.shape
top=divmod(np.flatnonzero(thresh)[0], sz[0])[::-1]
botton=divmod(np.flatnonzero(thresh)[-1], sz[0])[::-1]
thresh=thresh.transpose()
left=divmod(np.flatnonzero(thresh)[0], sz[1])
right=divmod(np.flatnonzero(thresh)[-1], sz[1])
print(top, botton, left, right, sep="\n")
# newimg=newimg[top[0]-1:botton[0]+1, left[0]-1:right[0]+1]
# newm=newm[top[0]-1:botton[0]+1, left[0]-1:right[0]+1]
greenRimg = greenScreen1(newimg,newm, top[1]-1, botton[1]+1, left[0] -1, right[0]+1)

end = time.time()
imageio.imwrite('output/greenRimg.png', greenRimg)
print("Time taken: ", end - start)