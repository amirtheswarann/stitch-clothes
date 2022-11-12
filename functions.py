import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure
import imutils

def removeWits(img):
    ret, binary_map = cv2.threshold(img,127,255,0)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if (areas[i] >= 2200)and areas[i] < 2400:
            result[labels == i + 1] = 255
    return result

def removeWits1(img):
    ret, binary_map = cv2.threshold(img,127,255,0)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if (areas[i] >= 350):
            result[labels == i + 1] = 255
    return result

def show2img(img1, img2):
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.show()
def getMask(img):
    # img = img[168:2500,844:5150]
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    A = lab[:,:,1]
    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = removeWits1(thresh)
    return thresh
    # blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)
    # mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)
    # result = img.copy()
    # result = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    # return result
    # result[:,:,3] = mask
    # return result

def greenScreen1(img, m, top,bottom,left,right):
    thresh = m
    blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)
    mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)
    result = img.copy()
    result = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    result[:,:,3] = mask
    return result[top:bottom,left:right]

def left_right(p1, p2):
    if p1[0] < p2[0]:
        return p1, p2
    else:
        return p2, p1

def rotate(img, angle):
    h, w = img.shape[:2]
    center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated
