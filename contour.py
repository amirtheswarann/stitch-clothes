import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure
import imutils
from functions import *

def center_point(cnts):
    # returns the center point of the contour
    M = cv2.moments(cnts)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def get_points(img,middleImage=False):
    ## we are finding contours in the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    threshed = removeWits(threshed)
    cnts = cv2.findContours(threshed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if middleImage:
        # for a middle image we need to find the for contours(required mark)
        cnts = cnts[0:4]
    else:
        # for first and last image we need onlt two marks (required marks)
        cnts = cnts[0:2]

    ## after finding contours we are finding the center of the contours
    cPoints = []
    for c in cnts:
        cPoints.append(center_point(c))
    return sorted(cPoints)