import time
start = time.time()
from cv2 import imwrite
from imageProcessing import *
from contour import *

#importing images
img1 = cv2.imread('input/snap1_Mark.jpg')
img2 = cv2.imread('input/snap2_Mark.jpg')

# croping to get only the required table
img1 = img1[168:2500,844:5150]
img2 = img2[168:2500,844:5150]

# getting the mask to remove the background
m1 = get_mask(img1)
m2 = get_mask(img2)

contour_img = cv2.imread('contours/crop1_0.jpg')
img1_points = get_points(img1, middleImage=False, match_contour_img=contour_img)
img2_points = get_points(img2, middleImage=False, match_contour_img=contour_img)


new_img, new_mask = stich_img(img1, img2, m1, m2, img1_points[0],img1_points[1], img2_points[0], img2_points[1])

green_removed_img = green_screen(new_img,new_mask)

imwrite('output/green_removed_img.png', green_removed_img)
end = time.time()

print("Time taken: ", end - start)