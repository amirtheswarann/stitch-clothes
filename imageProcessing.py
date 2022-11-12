import cv2
import numpy as np


def remove_wits(binary_map, min_area, max_area = None): 
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
    areas = stats[1:,cv2.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    if max_area is None:
        for i in range(0, nlabels - 1):
            if (areas[i] >= min_area): #keeps these wits
                result[labels == i + 1] = 255
    else:
        for i in range(0, nlabels - 1):
            if (areas[i] >= min_area) and (areas[i] < max_area): #keeps these wits
                result[labels == i + 1] = 255
    return result

def get_mask(img):
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    A = lab[:,:,1]
    thresh = cv2.threshold(A, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    thresh = remove_wits(thresh, 350)
    return thresh
