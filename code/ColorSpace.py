import cv2 as cv
import os

#Reading the Image
image = cv.imread(os.path.join('.','data','bird_02.jpg'))

#Covert the original image(BGR) to various color space like RGB,GRAY,HSV

#convert to RGB, using the "convert color" method
image_RGB = cv.cvtColor(image,cv.COLOR_BGR2RGB)

#convert to GRAY, using the "convert color" method
image_GRAY = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

#convert to HSV, using the "convert color" method
image_HSV = cv.cvtColor(image,cv.COLOR_BGR2HSV)


#Visualize the Image
cv.imshow("BGR-space",image)
cv.imshow("RGB-space",image_RGB)
cv.imshow("GRAY-space",image_GRAY)
cv.imshow("HSV-space",image_HSV)
cv.waitKey(0)