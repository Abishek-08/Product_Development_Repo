import cv2 as cv
import os

#Read the Image
rawImage = cv.imread(os.path.join('.','data','bear_01.jpg'))

#Convert the Image into Gray Image
gray_Image = cv.cvtColor(rawImage,cv.COLOR_BGR2GRAY)

#Doing the Threshold-1
ret,thresh = cv.threshold(rawImage,60,200,cv.THRESH_BINARY)

#Bluring the Image
thresh = cv.blur(thresh,(10,10))

#2nd Time doing the Threshold operaions
ret,thresh = cv.threshold(thresh,60,200,cv.THRESH_BINARY)

#visualize the Image
cv.imshow('RawImage',rawImage)
cv.imshow('GrayImage',gray_Image)
cv.imshow('Threshold-Image',thresh)
cv.waitKey(0)