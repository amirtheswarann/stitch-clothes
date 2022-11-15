import cv2
from imageProcessing import remove_wits, get_mask
import matplotlib.pyplot as plt
import statistics


from sklearn.cluster import KMeans
from collections import Counter

def center_point(cnts):
    # returns the center point of the contour
    M = cv2.moments(cnts)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)

def get_points(img,middleImage=False, match_contour_img=None):
    ## we are finding contours in the image

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    
    thresh = remove_wits(thresh, 2200, 2400) # ** this function will be very use if we are using lable
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    if match_contour_img is not None:
        match_contour_img_gray = cv2.cvtColor(match_contour_img, cv2.COLOR_BGR2GRAY)
        match_contour_img_thresh = cv2.threshold(match_contour_img_gray, 255, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        match_contour =  cv2.findContours(match_contour_img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

        mapp = {}
        for i in range(len(cnts)):
            match = cv2.matchShapes(cnts[i], match_contour[0], 1, 0.0)
            if match < 0.1:
                mapp[i] = cv2.matchShapes(cnts[i], match_contour[0], 1, 0.0)   
        cnts_new = []
        arr = sorted(mapp, key=mapp.get)
        for i in arr:
            cnts_new.append(cnts[i])
        cnts = cnts_new
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 255, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        thresh = remove_wits(thresh, 2200, 2400)
        cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

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


def keep_only_major(img):
    gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    plt.show()

    thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    # thresh = cv2.threshold(gray, 0, 127, cv2.THRESH_OTSU)[1]
    thresh = remove_wits(thresh,500)
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # show contour 
    # cv2.drawContours(img, cnts, -1, (0,255,0), 3)
     ##
    # plt.imshow(img)
    # plt.show()
    plt.imshow(thresh)
    plt.show()
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[:,:,3] = thresh
    plt.imshow(img)
    plt.show()


def keep_only_major_histogram(img):
    blue = img[:,:,0]
    green = img[:,:,1]
    red = img[:,:,2]

    blue = blue.flatten()
    green = green.flatten()
    red = red.flatten()
    #get min and max value for threshold using standard deviation
    blue_mean = blue.mean()
    blue_std = blue.std()
    blue_min = blue_mean - blue_std
    blue_max = blue_mean + blue_std

    green_mean = green.mean()
    green_std = green.std()
    green_min = green_mean - green_std
    green_max = green_mean + green_std

    red_mean = red.mean()
    red_std = red.std()

    
    #low =  (statistics.median_high(blue), statistics.median_high(green), statistics.median_high(green))
    low = (blue.mean()-blue.std(),green.mean()-green.std(),red.mean()-red.std())
    #low = (blue_mean,green_mean,red_mean)
    high = (blue.mean()+blue.std()**2,green.mean()+green.std()**2,red.mean()+red.std()**2)
    mask = cv2.inRange(img, low, high)
    mask = remove_wits(mask, 600)
    mask = ~mask
    plt.imshow(mask)
    plt.show()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img[:,:,3] = mask
    plt.imshow(img)
    plt.show()
