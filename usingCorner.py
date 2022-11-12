from functions import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
img1 = cv2.imread('input/snap1.jpg')
img2 = cv2.imread('input/snap2.jpg')
img1 = img1[168:2500,844:5150]
img2 = img2[168:2500,844:5150]

# get keypoints and descriptors using goodFeaturesToTrack
def getKeypointsAndDescriptors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    # corners = np.int0(corners)
    kp = []
    for i in corners:
        x, y = i.ravel()
        kp.append(cv2.KeyPoint(x, y, 1))
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.compute(img, kp)
    return kp, des

kp1, des1 = getKeypointsAndDescriptors(img1)
kp2, des2 = getKeypointsAndDescriptors(img2)

# get matches
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

# estimateAffinePartial2D
src_pts = np.float32([ kp1[m[0].queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([ kp2[m[0].trainIdx].pt for m in good ]).reshape(-1,1,2)
M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
matchesMask = mask.ravel().tolist()

# get the corners of the image
h, w = img1.shape[:2]
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)

# get the angle of rotation
p1, p2 = left_right(dst[0][0], dst[1][0])
angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi

# rotate the image
img2 = rotate(img2, angle)
