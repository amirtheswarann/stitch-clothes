import time
start = time.time()
from cv2 import imwrite
from imageProcessing import *
from contour import *

img1 = cv2.imread('input/snap1_Mark.jpg')
img2 = cv2.imread('input/snap2_Mark.jpg')

# croping to get only the required table
img1 = img1[168:2500,844:5150]
img2 = img2[168:2500,844:5150]
m1 = get_mask(img1)
m2 = get_mask(img2)

im1points = get_points(img1, middleImage=False)
im2points = get_points(img2, middleImage=False)
# print(im1points,im2points)

newimg, newm = stich_img(img1, img2, m1, m2, im1points[0],im1points[1], im2points[0], im2points[1])

greenRimg = greenScreen1(newimg,newm)
imwrite('output/greenRimg.png', greenRimg)
end = time.time()

print("Time taken: ", end - start)