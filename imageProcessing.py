import cv2
import numpy as np
import skimage.exposure
import matplotlib.pyplot as plt

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
    plt.imshow(thresh)
    plt.show()
    return thresh

def get_angle(p1, p2):
    yDiff = p2[0] - p1[0]
    xDiff = p2[1] - p1[1]
    return 90 - np.degrees(np.arctan2(yDiff, xDiff))

def rotate_bound(image, angle, center): # rotates the image around the center by the specified angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def find_endpoints(binary_img):
    # find the end points of the image
    top = 0
    bottom = binary_img.shape[0]
    left = 0
    right = binary_img.shape[1]
    for i in range(binary_img.shape[0]):
        if np.sum(binary_img[i,:]) > 0:
            top = i
            break
    for i in range(binary_img.shape[0]-1, -1, -1):
        if np.sum(binary_img[i,:]) > 0:
            bottom = i
            break
    for i in range(binary_img.shape[1]):
        if np.sum(binary_img[:,i]) > 0:
            left = i
            break
    for i in range(binary_img.shape[1]-1, -1, -1):
        if np.sum(binary_img[:,i]) > 0:
            right = i
            break
    return top, bottom, left, right

def stich_img(img1,img2, m1, m2, l1, r1, l2, r2):# m1 and m2 are the masks # l1 r1, l2, r2 are the contour points
    angle1 = get_angle(l1, r1)
    rotated_img1 = rotate_bound(img1, angle1, l1)
    rotated_img1 = rotated_img1[0:l1[1], 0:rotated_img1.shape[1]]

    m1 = rotate_bound(m1, angle1, l1)
    
    # removing image below the point 
    m1 = m1[0:l1[1], 0:m1.shape[1]]
    angle2 = get_angle(l2, r2)
    rotated_img2 = rotate_bound(img2, angle2, l2)
    rotated_img2 = rotated_img2[l2[1]:rotated_img2.shape[0], 0:rotated_img2.shape[1]]

    m2 = rotate_bound(m2, angle2, l2)

    # removing image above the point
    m2 = m2[l2[1]:m2.shape[0], 0:m2.shape[1]]
    cv2.imwrite('output/m21.jpg', m2)
    diff = l1[0] - l2[0]
    black1 = np.zeros((rotated_img1.shape[0], abs(diff), 3), dtype=np.uint8)
    mm1 = black1[:,:,0]
    black2 = np.zeros((rotated_img2.shape[0], abs(diff), 3), dtype=np.uint8)
    mm2 = black2[:,:,0]

    # adding black to the left to match the points
    # adding black to the right to match the width while concatenating
    if diff > 0:
        rotated_img2 = np.concatenate((black2, rotated_img2), axis=1)
        m2 = np.concatenate((mm2, m2), axis=1)
        rotated_img1 = np.concatenate((rotated_img1, black1), axis=1)
        m1 = np.concatenate((m1, mm1), axis=1)
    elif diff < 0:
        rotated_img1 = np.concatenate((black1, rotated_img1), axis=1)
        m1 = np.concatenate((mm1, m1), axis=1)
        rotated_img2 = np.concatenate((rotated_img2, black2), axis=1)
        m2 = np.concatenate((m2, mm2), axis=1)

    new_img = np.concatenate((rotated_img1, rotated_img2), axis=0)
    new_mask = np.concatenate((m1, m2), axis=0)
    top , bottom, left, right = find_endpoints(new_mask)

    new_img = new_img[top:bottom, left:right]
    new_mask = new_mask[top:bottom, left:right]

    return new_img, new_mask

def green_screen(img, thresh, green2alpha = True):
    blur = cv2.GaussianBlur(thresh, (0,0), sigmaX=5, sigmaY=5, borderType = cv2.BORDER_DEFAULT)
    mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5,255), out_range=(0,255)).astype(np.uint8)
    result = img.copy()
    if green2alpha:
        result = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        result[:,:,3] = mask
        return result
    else:
        mask =  ~mask
        result[:,:,0] = cv2.bitwise_or(img[:,:,0],mask)
        result[:,:,1] = cv2.bitwise_or(img[:,:,1],mask)
        result[:,:,2] = cv2.bitwise_or(img[:,:,2],mask)
        # result[:,:,0] = ~mask
        return result